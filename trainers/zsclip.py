import torch
import torch.nn as nn
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer
from tqdm import tqdm

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from .retrieval import (
    finalize_image_to_text_recall,
    finalize_text_to_image_recall_from_topk,
    init_text_to_image_topk,
    normalize_recall_ks,
    print_retrieval_results,
    update_image_to_text_hits,
    update_text_to_image_topk,
    write_retrieval_results,
)

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "NIHCXR": "a chest x-ray showing {}.",
}


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits

    @torch.no_grad()
    def evaluate_retrieval(self):
        direction = self.cfg.RETRIEVAL.DIRECTION
        recall_ks = normalize_recall_ks(self.cfg.RETRIEVAL.RECALL_KS)
        num_classes = self.text_features.size(0)

        i2t_hits = {f"R@{k}": 0.0 for k in recall_ks}
        i2t_total = 0
        max_k = max(recall_ks)
        max_k_img = min(max_k, len(self.dm.dataset.test))
        if max_k_img <= 0:
            raise ValueError("Retrieval evaluation requires at least one test image")
        best_scores, best_labels = init_text_to_image_topk(num_classes, max_k_img)
        class_counts = torch.zeros(num_classes, dtype=torch.long)
        num_test_images = 0

        self.set_model_mode("eval")
        for batch in tqdm(self.test_loader, ncols=80, desc="retrieval"):
            image = batch["img"].to(self.device)
            label = batch["label"].to(self.device)
            logits = self.model_inference(image)
            logits = logits.cpu()
            labels = label.cpu()
            num_test_images += labels.numel()

            if direction in ["image_to_text", "both"]:
                i2t_hits, i2t_total = update_image_to_text_hits(
                    i2t_hits, i2t_total, logits, labels, recall_ks
                )

            if direction in ["text_to_image", "both"]:
                best_scores, best_labels = update_text_to_image_topk(
                    best_scores, best_labels, logits, labels
                )
                class_counts += torch.bincount(labels, minlength=num_classes)

        direction_results = {}
        if direction in ["image_to_text", "both"]:
            direction_results["image_to_text"] = finalize_image_to_text_recall(
                i2t_hits, i2t_total, recall_ks
            )

        if direction in ["text_to_image", "both"]:
            t2i_recalls, valid_q, skipped_q = finalize_text_to_image_recall_from_topk(
                best_labels, class_counts, recall_ks, num_classes
            )
            t2i_recalls["valid_class_queries"] = valid_q
            t2i_recalls["skipped_class_queries"] = skipped_q
            direction_results["text_to_image"] = t2i_recalls

        retrieval_results = {
            "dataset": self.cfg.DATASET.NAME,
            "trainer": self.cfg.TRAINER.NAME,
            "prompt_mode": "zero-shot-hard-prompt",
            "direction": direction,
            "recall_ks": recall_ks,
            "num_test_images": int(num_test_images),
            "num_candidate_texts": int(num_classes),
            "results": direction_results,
        }

        print_retrieval_results(retrieval_results)
        output_path = write_retrieval_results(
            self.cfg.OUTPUT_DIR, self.cfg.RETRIEVAL.RESULTS_FILE, retrieval_results
        )
        print(f"Saved retrieval results to {output_path}")


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model
