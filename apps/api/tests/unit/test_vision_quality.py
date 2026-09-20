"""화질 지표 단위 테스트.

실제 영상 없이 합성 이미지로 각 지표의 단조성을 검증한다. 절대값은 입력에 따라 달라지므로
"흐리면 값이 내려간다" 같은 방향성만 확정한다.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from app.pipeline.vision import quality
from app.pipeline.vision.quality import (
    colorfulness,
    compute_metrics,
    exposure_stats,
    noise_sigma,
    readability_score,
    sharpness_lapvar,
    sharpness_tenengrad,
    to_gray,
)
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]


def _blur(image: BgrImage, sigma: float) -> BgrImage:
    """가우시안 블러. cv2 스텁이 넓은 dtype을 돌려주므로 uint8로 좁혀서 반환한다."""
    return np.asarray(cv2.GaussianBlur(image, (0, 0), sigma), np.uint8)


@pytest.fixture
def checkerboard() -> BgrImage:
    """선명한 에지가 많은 이미지.

    1픽셀 격자(Nyquist)는 3x3 Laplacian/Sobel 응답이 0이 되어 버리므로, 실제 사진의
    에지에 가깝게 8픽셀 블록으로 만든다.
    """
    block = 8
    rows = (np.arange(240) // block) % 2
    cols = (np.arange(320) // block) % 2
    board = (rows[:, None] ^ cols[None, :]).astype(np.uint8) * 255
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR).astype(np.uint8)


@pytest.fixture
def flat_gray() -> BgrImage:
    """완전히 균일한 무채색 이미지."""
    return np.full((240, 320, 3), 128, dtype=np.uint8)


def test_blur_lowers_sharpness(checkerboard: BgrImage) -> None:
    sharp = sharpness_lapvar(to_gray(checkerboard))
    blurred = sharpness_lapvar(to_gray(_blur(checkerboard, 3.0)))
    assert blurred < sharp


def test_blur_lowers_tenengrad(checkerboard: BgrImage) -> None:
    sharp = sharpness_tenengrad(to_gray(checkerboard))
    blurred = sharpness_tenengrad(to_gray(_blur(checkerboard, 3.0)))
    assert blurred < sharp


def test_flat_image_has_near_zero_sharpness(flat_gray: BgrImage) -> None:
    assert sharpness_lapvar(to_gray(flat_gray)) == pytest.approx(0.0, abs=1e-6)


def test_noise_sigma_increases_with_noise(flat_gray: BgrImage) -> None:
    rng = np.random.default_rng(0)
    clean = noise_sigma(to_gray(flat_gray))
    noisy_image = np.clip(
        flat_gray.astype(np.float32) + rng.normal(0.0, 10.0, flat_gray.shape), 0, 255
    ).astype(np.uint8)
    noisy = noise_sigma(to_gray(noisy_image))
    assert noisy > clean
    # Immerkaer 추정치는 참값 근방에 들어와야 한다(합성 가우시안 sigma=10).
    assert 6.0 < noisy < 14.0


def test_exposure_stats_detects_clipping() -> None:
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    white = np.full((100, 100, 3), 255, dtype=np.uint8)

    luma_dark, low_dark, high_dark, _ = exposure_stats(to_gray(black))
    luma_bright, low_bright, high_bright, _ = exposure_stats(to_gray(white))

    assert luma_dark == pytest.approx(0.0)
    assert low_dark == pytest.approx(1.0)
    assert high_dark == pytest.approx(0.0)
    assert luma_bright == pytest.approx(1.0)
    assert low_bright == pytest.approx(0.0)
    assert high_bright == pytest.approx(1.0)


def test_colorfulness_gray_is_near_zero(flat_gray: BgrImage) -> None:
    assert colorfulness(flat_gray) == pytest.approx(0.0, abs=1e-3)


def test_colorfulness_rises_with_saturation(flat_gray: BgrImage) -> None:
    colorful = flat_gray.copy()
    colorful[:, :160, 2] = 255  # 왼쪽 절반을 빨강으로
    colorful[:, 160:, 0] = 255  # 오른쪽 절반을 파랑으로
    assert colorfulness(colorful) > colorfulness(flat_gray)


def test_compute_metrics_clamps_textness(flat_gray: BgrImage) -> None:
    assert compute_metrics(flat_gray, textness=5.0).textness == pytest.approx(1.0)
    assert compute_metrics(flat_gray, textness=-2.0).textness == pytest.approx(0.0)


def test_readability_score_prefers_sharper_frame(checkerboard: BgrImage) -> None:
    sharp_metrics = compute_metrics(checkerboard)
    blurred_metrics = compute_metrics(_blur(checkerboard, 4.0))
    reference = sharp_metrics.sharpness_lapvar

    sharp_score = readability_score(sharp_metrics, sharpness_ref=reference)
    blurred_score = readability_score(blurred_metrics, sharpness_ref=reference)
    assert sharp_score > blurred_score


def test_readability_score_is_bounded(flat_gray: BgrImage) -> None:
    metrics = compute_metrics(flat_gray, textness=1.0)
    score = readability_score(metrics, sharpness_ref=1.0)
    assert 0.0 <= score <= 1.0


def test_readability_score_text_weight_shifts_result(flat_gray: BgrImage) -> None:
    with_text = compute_metrics(flat_gray, textness=1.0)
    without_text = compute_metrics(flat_gray, textness=0.0)
    reference = 100.0

    assert readability_score(with_text, sharpness_ref=reference, text_weight=0.9) > (
        readability_score(without_text, sharpness_ref=reference, text_weight=0.9)
    )


def test_noise_sigma_handles_tiny_image() -> None:
    tiny = np.zeros((2, 2), dtype=np.uint8)
    assert quality.noise_sigma(tiny) == 0.0
