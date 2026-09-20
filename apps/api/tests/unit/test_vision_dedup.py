"""임베더 · 샷 분할 · 중복 제거 단위 테스트."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from app.pipeline.vision import embed as embed_module
from app.pipeline.vision.dedup import (
    adaptive_scene_threshold,
    adjacent_similarities,
    cosine_similarity_matrix,
    drop_near_duplicate_shots,
    segment_shots,
    select_diverse_budget,
)
from app.pipeline.vision.embed import (
    PerceptualEmbedder,
    build_embedder,
    hsv_histogram,
    phash_bits,
)
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]


def _embed_scene(seed: int) -> NDArray[np.float32]:
    """장면 하나의 임베딩(1, D)."""
    return PerceptualEmbedder().embed([_scene(seed)])


def _sequence_with_adjacent_similarities(similarities: list[float]) -> NDArray[np.float32]:
    """인접 유사도가 정확히 주어진 값이 되는 단위벡터 수열을 만든다.

    2차원 평면에서 각도를 누적 회전시킨다. cos(θ_i) = similarities[i] 이므로 i번째와
    i+1번째의 내적이 그대로 지정값이 된다. 길이는 len(similarities) + 1.
    """
    angles = np.concatenate(
        [[0.0], np.cumsum(np.arccos(np.clip(np.array(similarities, dtype=np.float64), -1.0, 1.0)))]
    )
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def _scene(seed: int, height: int = 180, width: int = 240) -> BgrImage:
    """seed마다 뚜렷하게 다른 합성 장면."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = tuple(int(value) for value in rng.integers(20, 220, size=3))
    for _ in range(10):
        x1, y1 = int(rng.integers(0, width - 40)), int(rng.integers(0, height - 40))
        x2, y2 = x1 + int(rng.integers(20, 90)), y1 + int(rng.integers(20, 70))
        color = tuple(int(value) for value in rng.integers(0, 255, size=3))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    return image


def _jitter(image: BgrImage, shift: int = 2, seed: int = 0) -> BgrImage:
    """핸드헬드 흔들림 정도의 미세한 평행이동."""
    rng = np.random.default_rng(seed)
    matrix = np.array(
        [
            [1.0, 0.0, float(rng.integers(-shift, shift + 1))],
            [0.0, 1.0, float(rng.integers(-shift, shift + 1))],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT
    ).astype(np.uint8)


# --------------------------------------------------------------------------- 임베더


def test_phash_length_and_values() -> None:
    bits = phash_bits(_scene(1))
    assert bits.shape == (63,)  # 8x8 - DC
    assert set(np.unique(bits)).issubset({-1.0, 1.0})


def test_hsv_histogram_is_normalized() -> None:
    hist = hsv_histogram(_scene(1))
    assert hist.sum() == pytest.approx(1.0, abs=1e-5)


def test_embeddings_are_l2_normalized() -> None:
    embedder = PerceptualEmbedder()
    matrix = embedder.embed([_scene(1), _scene(2), _scene(3)])
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_identical_frames_have_similarity_one() -> None:
    image = _scene(4)
    matrix = PerceptualEmbedder().embed([image, image.copy()])
    assert float(matrix[0] @ matrix[1]) == pytest.approx(1.0, abs=1e-5)


def test_different_scenes_are_less_similar_than_jittered_same_scene() -> None:
    base = _scene(5)
    embedder = PerceptualEmbedder()
    matrix = embedder.embed([base, _jitter(base, seed=1), _scene(6)])

    same_scene = float(matrix[0] @ matrix[1])
    other_scene = float(matrix[0] @ matrix[2])
    assert same_scene > other_scene


def test_build_embedder_perceptual() -> None:
    assert build_embedder("perceptual").name.startswith("perceptual")


def test_build_embedder_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="알 수 없는 임베더"):
        build_embedder("nope")


@pytest.mark.parametrize("error", [ImportError("no torch"), OSError("gated repo 403")])
def test_build_embedder_auto_falls_back_when_dinov3_unavailable(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def _raise(**_: object) -> None:
        raise error

    monkeypatch.setattr(embed_module, "Dinov3Embedder", _raise)
    assert build_embedder("auto").name.startswith("perceptual")


def test_build_embedder_dinov3_does_not_swallow_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(**_: object) -> None:
        raise OSError("gated repo 403")

    monkeypatch.setattr(embed_module, "Dinov3Embedder", _raise)
    with pytest.raises(OSError, match="gated"):
        build_embedder("dinov3")


# ----------------------------------------------------------------------- 유사도 유틸


def test_cosine_similarity_matrix_diagonal_is_one() -> None:
    matrix = PerceptualEmbedder().embed([_scene(1), _scene(2)])
    sims = cosine_similarity_matrix(matrix)
    assert np.allclose(np.diag(sims), 1.0, atol=1e-5)


def test_adjacent_similarities_length() -> None:
    matrix = PerceptualEmbedder().embed([_scene(i) for i in range(4)])
    assert adjacent_similarities(matrix).shape == (3,)


def test_adjacent_similarities_empty_for_single_frame() -> None:
    matrix = PerceptualEmbedder().embed([_scene(1)])
    assert adjacent_similarities(matrix).shape == (0,)


# ------------------------------------------------------------------------ 샷 분할


def test_identical_frames_form_one_shot() -> None:
    image = _scene(7)
    frames = [image.copy() for _ in range(6)]
    matrix = PerceptualEmbedder().embed(frames)

    shots = segment_shots(
        matrix,
        [float(index) for index in range(6)],
        list(range(6)),
        similarity_threshold=0.88,
        min_shot_gap_sec=1.0,
    )
    assert len(shots) == 1
    assert shots[0].frame_indices == list(range(6))


def test_scene_change_splits_shots() -> None:
    frames = [_scene(8)] * 4 + [_scene(9)] * 4
    matrix = PerceptualEmbedder().embed(frames)

    shots = segment_shots(
        matrix,
        [float(index) for index in range(8)],
        list(range(8)),
        similarity_threshold=0.90,
        min_shot_gap_sec=1.0,
    )
    assert len(shots) == 2
    assert shots[0].frame_indices == [0, 1, 2, 3]
    assert shots[1].frame_indices == [4, 5, 6, 7]


def test_min_shot_gap_prevents_oversegmentation() -> None:
    """장면이 매 프레임 바뀌어도 최소 샷 길이 안에서는 새 샷을 열지 않는다."""
    frames = [_scene(index) for index in range(6)]
    matrix = PerceptualEmbedder().embed(frames)

    shots = segment_shots(
        matrix,
        [float(index) for index in range(6)],
        list(range(6)),
        similarity_threshold=0.99,
        min_shot_gap_sec=10.0,  # 전체 길이보다 긴 최소 간격
    )
    assert len(shots) == 1


def test_segment_shots_empty_input() -> None:
    assert segment_shots(np.zeros((0, 4), dtype=np.float32), [], []) == []


def test_segment_shots_rejects_length_mismatch() -> None:
    matrix = PerceptualEmbedder().embed([_scene(1), _scene(2)])
    with pytest.raises(ValueError, match="길이가 일치"):
        segment_shots(matrix, [0.0], [0, 1])


def test_shot_timestamps_are_ordered() -> None:
    frames = [_scene(10)] * 3 + [_scene(11)] * 3
    matrix = PerceptualEmbedder().embed(frames)
    shots = segment_shots(
        matrix,
        [float(index) for index in range(6)],
        list(range(6)),
        similarity_threshold=0.90,
        min_shot_gap_sec=1.0,
    )
    for shot in shots:
        assert shot.start_sec <= shot.end_sec
    assert shots[0].end_sec <= shots[1].start_sec


# ------------------------------------------------------------------- 중복 샷 제거


def test_revisited_scene_is_dropped() -> None:
    """같은 장소로 돌아온 샷(A, B, A')에서 A'가 제거되어야 한다."""
    scene_a, scene_b = _scene(12), _scene(13)
    matrix = PerceptualEmbedder().embed([scene_a, scene_b, _jitter(scene_a, seed=2)])

    kept = drop_near_duplicate_shots(matrix, similarity_threshold=0.95)
    assert kept == [0, 1]


def test_distinct_shots_all_kept() -> None:
    matrix = PerceptualEmbedder().embed([_scene(index) for index in range(14, 18)])
    assert drop_near_duplicate_shots(matrix, similarity_threshold=0.95) == [0, 1, 2, 3]


def test_drop_duplicates_empty_input() -> None:
    assert drop_near_duplicate_shots(np.zeros((0, 4), dtype=np.float32)) == []


def test_drop_duplicates_keeps_first_occurrence() -> None:
    image = _scene(19)
    matrix = PerceptualEmbedder().embed([image, image.copy(), image.copy()])
    assert drop_near_duplicate_shots(matrix, similarity_threshold=0.95) == [0]


# ------------------------------------------------- 예산 컷: 다양성 우선 (회귀)


def test_budget_keeps_distinct_scenes_over_high_scores() -> None:
    """점수 상위 K개를 남기면 잘 찍힌 한 장면이 슬롯을 독식한다.

    실측에서 정상 노출 장소 한 곳이 이 경로로 통째로 사라졌다. 같은 장면 3개와
    다른 장면 1개가 있고 예산이 2라면, 점수가 낮아도 다른 장면이 남아야 한다.
    """
    same = np.array([1.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0], dtype=np.float32)
    embeddings = np.vstack([same, same, same, other]).astype(np.float32)
    scores = [0.9, 0.85, 0.8, 0.2]

    kept = select_diverse_budget(embeddings, scores, budget=2)

    assert len(kept) == 2
    assert 3 in kept, "유사도가 먼 장면이 점수 때문에 탈락했다"


def test_budget_returns_everything_when_within_limit() -> None:
    embeddings = np.vstack([_embed_scene(seed) for seed in (1, 2, 3)])
    assert select_diverse_budget(embeddings, [0.5, 0.4, 0.3], budget=5) == [0, 1, 2]


# ------------------------------------------- 장면 임계 자동 보정 (회귀)


def test_adaptive_threshold_loosens_on_panning_footage() -> None:
    """패닝이 많은 영상에서는 고정 임계가 같은 장소를 과분할한다.

    인접 유사도가 대부분 0.75 근처이고 장면 전환에서만 0.3으로 떨어지는 경우,
    0.88을 그대로 쓰면 모든 프레임이 각자 샷이 된다.
    """
    adjacent = [0.75, 0.78, 0.30, 0.76, 0.77, 0.79, 0.74]
    panning = _sequence_with_adjacent_similarities(adjacent)
    assert adjacent_similarities(panning) == pytest.approx(adjacent, abs=1e-3)

    threshold = adaptive_scene_threshold(panning, default_threshold=0.88, max_cut_rate=0.25)

    # 같은 장소의 인접쌍(0.74~0.79)은 살리고 장면 전환(0.30)만 컷으로 남아야 한다.
    assert threshold == pytest.approx(0.74, abs=1e-3)
    cut_rate = float((adjacent_similarities(panning) < threshold).mean())
    assert cut_rate <= 0.25


def test_adaptive_threshold_keeps_default_on_static_footage() -> None:
    """삼각대 영상(인접 유사도 ~1.0)에서는 기존 동작이 유지되어야 한다."""
    static = np.vstack([_embed_scene(1) for _ in range(8)])
    threshold = adaptive_scene_threshold(static, default_threshold=0.88, max_cut_rate=0.25)
    assert threshold == pytest.approx(0.88)


def test_adaptive_threshold_needs_enough_samples() -> None:
    tiny = np.vstack([_embed_scene(1), _embed_scene(2)])
    assert adaptive_scene_threshold(tiny, default_threshold=0.88) == pytest.approx(0.88)
