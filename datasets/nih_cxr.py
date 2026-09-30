import csv
import glob
import os
import os.path as osp
import pickle
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class NIHCXR(DatasetBase):
    """NIH ChestX-ray14 as a single-label classification benchmark.

    ChestX-ray14 annotations can contain more than one finding per image.
    CoOp uses a single-label cross-entropy loss, so this adapter retains only
    images with exactly one finding in every split.
    """

    dataset_dir = "nih_cxr"
    metadata_file = "Data_Entry_2017.csv"
    train_list_file = "train_val_list.txt"
    test_list_file = "test_list.txt"
    classnames = [
        "atelectasis", "cardiomegaly", "effusion", "infiltration", "mass",
        "nodule", "pneumonia", "pneumothorax", "consolidation", "edema",
        "emphysema", "fibrosis", "pleural thickening", "hernia", "no finding",
    ]
    finding_to_label = {
        "Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2,
        "Infiltration": 3, "Mass": 4, "Nodule": 5, "Pneumonia": 6,
        "Pneumothorax": 7, "Consolidation": 8, "Edema": 9,
        "Emphysema": 10, "Fibrosis": 11, "Pleural_Thickening": 12,
        "Hernia": 13, "No Finding": 14,
    }

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        if osp.isfile(osp.join(root, self.metadata_file)):
            self.dataset_dir = root
        else:
            self.dataset_dir = osp.join(root, self.dataset_dir)

        self.split_path = osp.join(self.dataset_dir, "split_zhou_NIHCXR.json")
        self.split_fewshot_dir = osp.join(self.dataset_dir, "split_fewshot")
        self.metadata_path = osp.join(self.dataset_dir, self.metadata_file)
        self.train_list_path = osp.join(self.dataset_dir, self.train_list_file)
        self.test_list_path = osp.join(self.dataset_dir, self.test_list_file)
        self._check_required_files()
        mkdir_if_missing(self.split_fewshot_dir)

        if osp.exists(self.split_path):
            train, val, test = OxfordPets.read_split(
                self.split_path, self.dataset_dir
            )
        else:
            image_paths = self._index_images()
            annotations = self._read_annotations()
            trainval = self._read_split(
                self.train_list_path, annotations, image_paths
            )
            test = self._read_split(self.test_list_path, annotations, image_paths)
            train, val = self.split_trainval(trainval)
            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = osp.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )
            if osp.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(
            train, val, test, subsample=subsample
        )
        super().__init__(train_x=train, val=val, test=test)

    def _check_required_files(self):
        required_files = [
            self.dataset_dir,
            self.metadata_path,
            self.train_list_path,
            self.test_list_path,
        ]
        missing_files = [path for path in required_files if not osp.exists(path)]
        if missing_files:
            raise FileNotFoundError(
                "NIH CXR is incomplete. Missing required path(s): "
                + ", ".join(missing_files)
            )

    def _index_images(self):
        paths = glob.glob(osp.join(self.dataset_dir, "images*", "images", "*.png"))
        if not paths:
            raise RuntimeError(
                "No PNG images found under "
                f"{osp.join(self.dataset_dir, 'images*', 'images')}"
            )
        return {osp.basename(path): path for path in paths}

    def _read_annotations(self):
        annotations = {}
        with open(self.metadata_path, newline="") as file:
            for row in csv.DictReader(file):
                findings = row["Finding Labels"].split("|")
                if len(findings) != 1:
                    continue
                finding = findings[0]
                if finding in self.finding_to_label:
                    annotations[row["Image Index"]] = finding
        return annotations

    def _read_split(self, split_path, annotations, image_paths):
        items = []
        missing_images = []
        with open(split_path) as file:
            for line in file:
                image_name = line.strip()
                if not image_name or image_name not in annotations:
                    continue
                impath = image_paths.get(image_name)
                if impath is None:
                    missing_images.append(image_name)
                    continue
                finding = annotations[image_name]
                label = self.finding_to_label[finding]
                items.append(Datum(impath=impath, label=label, classname=self.classnames[label]))
        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} images listed in {split_path} were not found; "
                f"first missing image: {missing_images[0]}"
            )
        if not items:
            raise RuntimeError(f"No single-label images found in {split_path}")
        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        tracker = defaultdict(list)
        for item in trainval:
            tracker[item.label].append(item)

        rng = random.Random(1)
        train, val = [], []
        for items in tracker.values():
            rng.shuffle(items)
            n_val = max(1, round(len(items) * p_val))
            val.extend(items[:n_val])
            train.extend(items[n_val:])
        return train, val
