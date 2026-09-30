import unittest
import importlib.util

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from trainers.retrieval import (
        finalize_image_to_text_recall,
        finalize_text_to_image_recall_from_topk,
        image_to_text_recall_from_logits,
        init_text_to_image_topk,
        normalize_recall_ks,
        text_to_image_recall_from_logits,
        update_image_to_text_hits,
        update_text_to_image_topk,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class RetrievalMetricsTest(unittest.TestCase):
    def test_normalize_recall_ks(self):
        self.assertEqual(normalize_recall_ks([10, 1, 5, 5]), [1, 5, 10])
        with self.assertRaises(ValueError):
            normalize_recall_ks([0, 1])

    def test_image_to_text_recall_configurable_k(self):
        logits = torch.tensor(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.6, 0.4],
            ]
        )
        labels = torch.tensor([0, 1, 1])
        recalls = image_to_text_recall_from_logits(logits, labels, [1, 2])
        self.assertAlmostEqual(recalls["R@1"], 66.6666666667, places=4)
        self.assertAlmostEqual(recalls["R@2"], 100.0, places=4)

    def test_text_to_image_relevance_and_skip_behavior(self):
        logits = torch.tensor(
            [
                [0.90, 0.20, 0.10],
                [0.80, 0.30, 0.20],
                [0.10, 0.40, 0.30],
            ]
        )
        labels = torch.tensor([0, 0, 1])  # class 2 has no test image
        recalls, valid_q, skipped_q = text_to_image_recall_from_logits(
            logits, labels, [1, 2], num_classes=3
        )

        self.assertEqual(valid_q, 2)
        self.assertEqual(skipped_q, 1)
        self.assertAlmostEqual(recalls["R@1"], 100.0, places=4)
        self.assertAlmostEqual(recalls["R@2"], 100.0, places=4)

    def test_streaming_matches_full_metrics(self):
        recall_ks = [1, 2]
        batch1_logits = torch.tensor([[0.9, 0.1], [0.4, 0.6]])
        batch1_labels = torch.tensor([0, 1])
        batch2_logits = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
        batch2_labels = torch.tensor([0, 1])

        logits = torch.cat([batch1_logits, batch2_logits], dim=0)
        labels = torch.cat([batch1_labels, batch2_labels], dim=0)

        full_i2t = image_to_text_recall_from_logits(logits, labels, recall_ks)
        full_t2i, full_valid_q, full_skipped_q = text_to_image_recall_from_logits(
            logits, labels, recall_ks, num_classes=2
        )

        hits = {f"R@{k}": 0.0 for k in recall_ks}
        total = 0
        hits, total = update_image_to_text_hits(hits, total, batch1_logits, batch1_labels, recall_ks)
        hits, total = update_image_to_text_hits(hits, total, batch2_logits, batch2_labels, recall_ks)
        stream_i2t = finalize_image_to_text_recall(hits, total, recall_ks)

        best_scores, best_labels = init_text_to_image_topk(num_classes=2, max_k=2)
        class_counts = torch.zeros(2, dtype=torch.long)
        best_scores, best_labels = update_text_to_image_topk(best_scores, best_labels, batch1_logits, batch1_labels)
        class_counts += torch.bincount(batch1_labels, minlength=2)
        best_scores, best_labels = update_text_to_image_topk(best_scores, best_labels, batch2_logits, batch2_labels)
        class_counts += torch.bincount(batch2_labels, minlength=2)
        stream_t2i, stream_valid_q, stream_skipped_q = finalize_text_to_image_recall_from_topk(
            best_labels, class_counts, recall_ks, num_classes=2
        )

        self.assertEqual(full_valid_q, stream_valid_q)
        self.assertEqual(full_skipped_q, stream_skipped_q)
        self.assertEqual(full_i2t, stream_i2t)
        self.assertEqual(full_t2i, stream_t2i)


if __name__ == "__main__":
    unittest.main()
