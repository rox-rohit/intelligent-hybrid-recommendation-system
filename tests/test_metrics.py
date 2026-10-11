"""Unit tests for evaluation metrics (hand-computed ground truth)."""

import numpy as np

from src.evaluation.metrics import (
    catalog_coverage,
    evaluate_all,
    hit_rate_at_k,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# One user; truth = {101, 102}; recs ranked list of 5.
RECS = np.array([[101, 900, 102, 700, 800]])
TRUTH = {0: {101, 102}}


class TestPrecisionRecall:
    def test_precision_half(self):
        # 2 hits in top-5 -> 0.4
        assert precision_at_k(RECS, TRUTH, k=5) == 0.4

    def test_precision_at_2(self):
        # top-2 = [101, 900]: 1 hit -> 0.5
        assert precision_at_k(RECS, TRUTH, k=2) == 0.5

    def test_recall_full(self):
        # both relevant items appear -> recall 1.0
        assert recall_at_k(RECS, TRUTH, k=5) == 1.0

    def test_recall_at_2_half(self):
        # only 101 in top-2 -> 0.5
        assert recall_at_k(RECS, TRUTH, k=2) == 0.5

    def test_no_overlap_zero(self):
        recs = np.array([[1, 2, 3]])
        assert precision_at_k(recs, {0: {9}}, k=3) == 0.0
        assert recall_at_k(recs, {0: {9}}, k=3) == 0.0


class TestHitRate:
    def test_hit(self):
        assert hit_rate_at_k(RECS, TRUTH, k=5) == 1.0

    def test_miss(self):
        recs = np.array([[1, 2, 3]])
        assert hit_rate_at_k(recs, {0: {9}}, k=3) == 0.0

    def test_averaged_over_users(self):
        recs = np.array([[1, 2, 3], [4, 5, 6]])
        truth = {0: {1}, 1: {99}}  # user 0 hits, user 1 misses
        assert hit_rate_at_k(recs, truth, k=3) == 0.5


class TestNDCG:
    def test_perfect_ranking(self):
        recs = np.array([[101, 102, 1]])
        assert np.isclose(ndcg_at_k(recs, TRUTH, k=3), 1.0)

    def test_hand_computed(self):
        # recs = [101, 900, 102]: hits at positions 1 and 3
        # DCG  = 1/log2(2) + 1/log2(4) = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.6309   (both hits ranked first)
        assert np.isclose(ndcg_at_k(RECS, TRUTH, k=3), 1.5 / (1 + 1 / np.log2(3)))

    def test_worse_than_perfect(self):
        # both hits at the tail -> NDCG < 1
        recs = np.array([[900, 700, 101, 102, 1]])
        assert ndcg_at_k(recs, TRUTH, k=5) < 1.0
        assert ndcg_at_k(recs, TRUTH, k=5) > 0

    def test_zero_when_no_truth(self):
        assert ndcg_at_k(RECS, {}, k=5) == 0.0


class TestMAP:
    def test_hand_computed(self):
        # hits at ranks 1,3: AP = (1/1 + 2/3)/2 = 0.8333
        assert np.isclose(map_at_k(RECS, TRUTH, k=5), (1 + 2 / 3) / 2)

    def test_perfect(self):
        recs = np.array([[101, 102, 1]])
        assert np.isclose(map_at_k(recs, TRUTH, k=3), 1.0)


class TestCoverage:
    def test_coverage(self):
        recs = np.array([[1, 2], [2, 3]])
        assert catalog_coverage(recs, n_items=10) == 0.3  # {1,2,3} of 10

    def test_coverage_zero_items(self):
        assert catalog_coverage(np.zeros((1, 1), dtype=int), n_items=0) == 0.0


class TestEvaluateAll:
    def test_suite_keys(self):
        out = evaluate_all(RECS, TRUTH, k=5, n_items=1000)
        assert set(out) == {
            "precision@5", "recall@5", "hit_rate@5", "ndcg@5", "map@5", "coverage",
        }
        assert all(0 <= v <= 1.0001 for v in out.values())

    def test_user_keys_align_with_rows(self):
        # truth keyed by real user ids must align positionally with recs rows
        recs = np.array([[101], [102]])
        truth = {7: {101}, 8: {102}}
        assert precision_at_k(recs, truth, k=1) == 1.0
