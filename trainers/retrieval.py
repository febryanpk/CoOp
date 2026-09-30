import json
import os

import torch


def normalize_recall_ks(recall_ks):
    ks = []
    for k in recall_ks:
        k = int(k)
        if k <= 0:
            raise ValueError(f"Recall@k expects positive integers, but got {k}")
        ks.append(k)
    ks = sorted(set(ks))
    return ks


def image_to_text_recall_from_logits(logits, labels, recall_ks):
    recall_ks = normalize_recall_ks(recall_ks)
    max_k = min(max(recall_ks), logits.size(1))
    topk = torch.topk(logits, k=max_k, dim=1).indices
    labels = labels.view(-1, 1)

    recalls = {}
    for k in recall_ks:
        k_eff = min(k, max_k)
        correct = topk[:, :k_eff].eq(labels).any(dim=1)
        recalls[f"R@{k}"] = 100.0 * correct.float().mean().item()

    return recalls


def text_to_image_recall_from_logits(logits, labels, recall_ks, num_classes):
    recall_ks = normalize_recall_ks(recall_ks)
    max_k = min(max(recall_ks), logits.size(0))
    labels = labels.view(-1)

    recalls = {f"R@{k}": 0.0 for k in recall_ks}
    valid_queries = 0
    skipped_queries = 0

    for class_idx in range(num_classes):
        relevant = labels == class_idx
        if not relevant.any():
            skipped_queries += 1
            continue

        valid_queries += 1
        ranked_img_indices = torch.topk(logits[:, class_idx], k=max_k, dim=0).indices
        for k in recall_ks:
            k_eff = min(k, max_k)
            hit = relevant[ranked_img_indices[:k_eff]].any().item()
            recalls[f"R@{k}"] += float(hit)

    if valid_queries > 0:
        for key in recalls:
            recalls[key] = 100.0 * recalls[key] / valid_queries

    return recalls, valid_queries, skipped_queries


def update_image_to_text_hits(hits, total, logits, labels, recall_ks):
    recall_ks = normalize_recall_ks(recall_ks)
    max_k = min(max(recall_ks), logits.size(1))
    topk = torch.topk(logits, k=max_k, dim=1).indices
    labels = labels.view(-1, 1)
    total += labels.size(0)

    for k in recall_ks:
        k_eff = min(k, max_k)
        correct = topk[:, :k_eff].eq(labels).any(dim=1).sum().item()
        hits[f"R@{k}"] += float(correct)

    return hits, total


def finalize_image_to_text_recall(hits, total, recall_ks):
    recalls = {}
    for k in normalize_recall_ks(recall_ks):
        key = f"R@{k}"
        recalls[key] = 0.0 if total == 0 else 100.0 * hits[key] / total
    return recalls


def init_text_to_image_topk(num_classes, max_k):
    best_scores = torch.full((num_classes, max_k), -float("inf"))
    best_labels = torch.full((num_classes, max_k), -1, dtype=torch.long)
    return best_scores, best_labels


def update_text_to_image_topk(best_scores, best_labels, logits, labels):
    num_classes, max_k = best_scores.shape
    scores = logits.t()
    labels = labels.to(best_labels.device)
    expanded_labels = labels.unsqueeze(0).expand(num_classes, -1)

    merged_scores = torch.cat([best_scores, scores], dim=1)
    merged_labels = torch.cat([best_labels, expanded_labels], dim=1)

    top_indices = torch.topk(merged_scores, k=max_k, dim=1).indices
    best_scores = torch.gather(merged_scores, 1, top_indices)
    best_labels = torch.gather(merged_labels, 1, top_indices)
    return best_scores, best_labels


def finalize_text_to_image_recall_from_topk(best_labels, class_counts, recall_ks, num_classes):
    recall_ks = normalize_recall_ks(recall_ks)
    max_k = best_labels.size(1)

    recalls = {f"R@{k}": 0.0 for k in recall_ks}
    valid_queries = 0
    skipped_queries = 0

    for class_idx in range(num_classes):
        if class_counts[class_idx] <= 0:
            skipped_queries += 1
            continue

        valid_queries += 1
        class_top_labels = best_labels[class_idx]

        for k in recall_ks:
            k_eff = min(k, max_k)
            hit = (class_top_labels[:k_eff] == class_idx).any().item()
            recalls[f"R@{k}"] += float(hit)

    if valid_queries > 0:
        for key in recalls:
            recalls[key] = 100.0 * recalls[key] / valid_queries

    return recalls, valid_queries, skipped_queries


def write_retrieval_results(output_dir, filename, payload):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    return output_path


def print_retrieval_results(retrieval_results):
    recall_ks = retrieval_results["recall_ks"]
    direction_results = retrieval_results["results"]

    print("\n===== Retrieval evaluation =====")
    print(f"dataset: {retrieval_results['dataset']}")
    print(f"trainer: {retrieval_results['trainer']}")
    print(f"mode: {retrieval_results['prompt_mode']}")
    print(f"directions: {retrieval_results['direction']}")
    print("Recall@k:", ", ".join(str(k) for k in recall_ks))
    print("Relevance rule (text-to-image): class query is correct if at least one image from that class is in top-k.")

    for direction_name, values in direction_results.items():
        print(f"- {direction_name}")
        for k in recall_ks:
            key = f"R@{k}"
            print(f"  {key}: {values[key]:.2f}")
        if "valid_class_queries" in values:
            print(
                f"  valid_class_queries: {values['valid_class_queries']} "
                f"(skipped_without_test_images={values['skipped_class_queries']})"
            )
