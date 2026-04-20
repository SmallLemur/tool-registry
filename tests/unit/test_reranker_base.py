"""Unit tests for reranker/base._apply_ranking.

Pure helper — no deps on pymilvus/config. Converts LLM-returned ranked
indices into a reordered candidate list with rerank_score attached. Safety
nets the helper provides: dedup, invalid-index skip, missed-candidate
appending at score 0.0, truncation to limit.
"""

from app.reranker.base import _apply_ranking


def _candidates(n: int) -> list[dict]:
    return [{"id": f"c{i}", "content": f"cap {i}"} for i in range(n)]


def test_reorders_by_ranked_indices():
    """Given [0,1,2] candidates and LLM indices [2,0,1], result order is c2,c0,c1."""
    c = _candidates(3)
    result = _apply_ranking(c, [2, 0, 1], limit=10)
    assert [r["id"] for r in result] == ["c2", "c0", "c1"]


def test_rerank_score_decreases_by_rank():
    c = _candidates(3)
    result = _apply_ranking(c, [2, 0, 1], limit=10)
    # rank-0 score is higher than rank-1, rank-1 higher than rank-2
    assert result[0]["rerank_score"] > result[1]["rerank_score"]
    assert result[1]["rerank_score"] > result[2]["rerank_score"]


def test_missing_candidates_appended_at_score_zero():
    """If the LLM skips candidate 1, it should still be in the result at score 0."""
    c = _candidates(3)
    result = _apply_ranking(c, [0, 2], limit=10)
    ids = [r["id"] for r in result]
    assert ids == ["c0", "c2", "c1"]
    # The appended one has rerank_score 0.0
    assert result[-1]["rerank_score"] == 0.0


def test_limit_truncates_result():
    c = _candidates(5)
    result = _apply_ranking(c, [0, 1, 2, 3, 4], limit=2)
    assert len(result) == 2
    assert [r["id"] for r in result] == ["c0", "c1"]


def test_duplicate_indices_dedupe():
    c = _candidates(3)
    result = _apply_ranking(c, [0, 0, 1, 1, 2], limit=10)
    ids = [r["id"] for r in result]
    # Each original candidate should appear exactly once
    assert ids.count("c0") == 1
    assert ids.count("c1") == 1
    assert ids.count("c2") == 1


def test_out_of_range_indices_skipped():
    """Negative or out-of-range indices are ignored."""
    c = _candidates(2)
    result = _apply_ranking(c, [5, -1, 0], limit=10)
    # Only index 0 is valid; c1 gets appended as missing
    ids = [r["id"] for r in result]
    assert ids[0] == "c0"
    assert "c1" in ids


def test_non_int_indices_skipped():
    c = _candidates(2)
    result = _apply_ranking(c, [0, "bad", None, 1], limit=10)
    ids = [r["id"] for r in result]
    assert ids == ["c0", "c1"]


def test_non_list_indices_returns_original_order():
    """If the LLM returns something other than a list, fall back gracefully."""
    c = _candidates(3)
    result = _apply_ranking(c, "not a list", limit=10)  # type: ignore[arg-type]
    assert [r["id"] for r in result] == ["c0", "c1", "c2"]


def test_empty_candidates_returns_empty():
    assert _apply_ranking([], [0, 1], limit=10) == []


def test_input_candidates_not_mutated():
    """The helper copies each dict it keeps, so the caller's list stays clean."""
    c = _candidates(2)
    _apply_ranking(c, [0, 1], limit=10)
    # No rerank_score key polluted into the input
    assert "rerank_score" not in c[0]
