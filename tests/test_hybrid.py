"""Unit tests for the weighted hybrid recommender (synthetic fixtures)."""

import pytest

from src.models.hybrid import HybridRecommender, HybridWeights, minmax


@pytest.fixture
def scorers():
    """Deterministic component scorers over a tiny catalog (items 1-5)."""
    cf = lambda u: {1: 5.0, 2: 3.0, 3: 1.0} if u == 0 else {4: 2.0, 5: 1.0}
    content = lambda u: {2: 4.0, 4: 2.0, 1: 1.0}
    pop = lambda u: {5: 3.0, 3: 2.0, 1: 1.0}
    return cf, content, pop


class TestMinmax:
    def test_basic(self):
        out = minmax({1: 2.0, 2: 4.0, 3: 6.0})
        assert out == {1: 0.0, 2: 0.5, 3: 1.0}

    def test_empty(self):
        assert minmax({}) == {}

    def test_constant(self):
        assert minmax({1: 3.0, 2: 3.0}) == {1: 0.0, 2: 0.0}


class TestHybridWeights:
    def test_normalized(self):
        w = HybridWeights(cf=2, content=1, popularity=1).normalized()
        assert w.as_tuple() == (0.5, 0.25, 0.25)

    def test_zero_total(self):
        w = HybridWeights(0, 0, 0).normalized()
        assert w.as_tuple() == (1 / 3, 1 / 3, 1 / 3)


class TestHybridRecommender:
    def _make(self, scorers, weights=None):
        cf, content, pop = scorers
        return HybridRecommender(cf, content, pop, weights)

    def test_scores_union_of_components(self, scorers):
        hy = self._make(scorers)
        scores = hy.score_items(0)
        assert set(scores) == {1, 2, 3, 4, 5}

    def test_seen_items_excluded(self, scorers):
        hy = self._make(scorers)
        scores = hy.score_items(0, seen_items={1, 2})
        assert 1 not in scores and 2 not in scores

    def test_candidate_restriction(self, scorers):
        hy = self._make(scorers)
        scores = hy.score_items(0, candidate_items={3, 5})
        assert set(scores) == {3, 5}

    def test_dominant_component_wins(self, scorers):
        # popularity weight 1.0: item 5 (top pop) must rank first for user 0
        hy = self._make(scorers, HybridWeights(cf=0, content=0, popularity=1))
        recs = hy.recommend(0, k=1)
        assert recs[0][0] == 5

    def test_cf_only_matches_cf_ranking(self, scorers):
        hy = self._make(scorers, HybridWeights(cf=1, content=0, popularity=0))
        recs = hy.recommend(0, k=3)
        assert [i for i, _ in recs] == [1, 2, 3]

    def test_missing_component_scores_zero(self, scorers):
        # item 4 has no CF or pop score for user 0 -> only content lifts it
        hy = self._make(scorers, HybridWeights(cf=0, content=1, popularity=0))
        scores = hy.score_items(0)
        assert scores[5] == 0.0  # no content score for 5

    def test_recommend_respects_k(self, scorers):
        hy = self._make(scorers)
        assert len(hy.recommend(0, k=3)) == 3

    def test_explain_dominant(self, scorers):
        hy = self._make(scorers, HybridWeights(cf=1, content=0, popularity=0))
        exp = hy.explain(0, 1)
        assert exp["dominant"] == "collaborative"
        assert isinstance(exp["reason"], str)

    def test_explain_item_outside_all(self, scorers):
        hy = self._make(scorers)
        exp = hy.explain(0, 999)  # unknown item -> all contributions 0
        assert set(exp["contributions"].values()) == {0.0}

    def test_no_scorers_at_all(self):
        hy = HybridRecommender()
        assert hy.recommend(0, k=5) == []

    def test_weights_normalized_on_construction(self, scorers):
        hy = self._make(scorers, HybridWeights(cf=10, content=0, popularity=0))
        # unnormalized weights must not blow up scores
        scores = hy.score_items(0)
        assert all(0.0 <= v <= 1.0 + 1e-9 for v in scores.values())
