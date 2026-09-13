import numpy as np

from openpi.policies.libero_victr_retrieval import linear_progress
from openpi.policies.libero_victr_retrieval import retrieve_indices


def test_linear_progress_includes_exact_endpoints():
    np.testing.assert_array_equal(linear_progress(1), [1.0])
    np.testing.assert_allclose(linear_progress(5), [0.0, 0.25, 0.5, 0.75, 1.0])


def test_three_retrieval_metrics_are_demo_diverse():
    embeddings = np.asarray(
        [[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        dtype=np.float32,
    )
    progress = np.asarray([0.0, 0.5, 0.3, 0.7, 0.4, 0.9], dtype=np.float32)
    demos = np.repeat(np.arange(3, dtype=np.int32), 2)
    for metric in ("vision", "progress", "vision_progress"):
        indices, scores = retrieve_indices(
            metric=metric,
            context_demo_indices=demos,
            k=3,
            context_embeddings=embeddings,
            query_embedding=np.asarray([0.0, 0.0], dtype=np.float32),
            context_progress=progress,
            query_progress=0.45,
        )
        assert indices.shape == scores.shape == (3,)
        assert len(np.unique(demos[indices])) == 3
        assert np.all(scores[:-1] <= scores[1:])


def test_fusion_matches_viktr_per_query_zscore_sum():
    embeddings = np.asarray([[0.0], [1.0], [3.0]], dtype=np.float32)
    progress = np.asarray([0.9, 0.5, 0.0], dtype=np.float32)
    demos = np.arange(3, dtype=np.int32)
    indices, scores = retrieve_indices(
        metric="vision_progress",
        context_demo_indices=demos,
        k=3,
        context_embeddings=embeddings,
        query_embedding=np.asarray([0.0], dtype=np.float32),
        context_progress=progress,
        query_progress=0.0,
    )
    dino = np.asarray([0.0, 1.0, 3.0], dtype=np.float32)
    value = progress
    expected = (dino - dino.mean()) / (dino.std() + 1e-8)
    expected += (value - value.mean()) / (value.std() + 1e-8)
    np.testing.assert_array_equal(indices, np.argsort(expected, kind="stable"))
    np.testing.assert_allclose(scores, expected[indices])
