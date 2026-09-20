"""미니 ISP 단위 테스트."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from app.pipeline.vision.isp import (
    EnhanceConfig,
    adaptive_gamma,
    clahe_luminance,
    enhance_for_vlm,
    rescue_exposure,
    unsharp_mask,
    white_balance_shades_of_gray,
)
from app.pipeline.vision.quality import compute_metrics, noise_sigma, to_gray
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]


def _blur(image: BgrImage, sigma: float) -> BgrImage:
    """가우시안 블러. cv2 스텁이 넓은 dtype을 돌려주므로 uint8로 좁혀서 반환한다."""
    return np.asarray(cv2.GaussianBlur(image, (0, 0), sigma), np.uint8)


def _textured(height: int = 240, width: int = 320, seed: int = 7) -> BgrImage:
    """구조가 있는 합성 프레임. 완전 균일 이미지에서는 WB/감마 판정이 퇴화한다."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 110, dtype=np.uint8)
    for _ in range(20):
        x1, y1 = int(rng.integers(0, width - 40)), int(rng.integers(0, height - 40))
        x2, y2 = x1 + int(rng.integers(20, 80)), y1 + int(rng.integers(20, 60))
        color = tuple(int(value) for value in rng.integers(30, 230, size=3))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    return image


def _blue_cast(image: BgrImage) -> BgrImage:
    """파란 채널을 키운 색온도 편향 프레임."""
    cast = image.astype(np.float32)
    cast[:, :, 0] *= 1.7
    cast[:, :, 2] *= 0.7
    return np.clip(cast, 0, 255).astype(np.uint8)


def test_white_balance_reduces_channel_imbalance() -> None:
    image = _blue_cast(_textured())
    before = image.reshape(-1, 3).mean(axis=0)
    balanced, gains = white_balance_shades_of_gray(image)
    after = balanced.reshape(-1, 3).mean(axis=0)

    assert float(after.std()) < float(before.std())
    assert gains[0] < 1.0 < gains[2]  # 파랑은 줄이고 빨강은 올린다


def test_white_balance_respects_gain_cap() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 250  # 파랑만 존재
    _, gains = white_balance_shades_of_gray(image, max_gain=2.0)
    assert all(0.5 - 1e-6 <= gain <= 2.0 + 1e-6 for gain in gains)


def test_adaptive_gamma_brightens_dark_frame() -> None:
    dark = (_textured().astype(np.float32) * 0.2).astype(np.uint8)
    brightened, gamma = adaptive_gamma(dark, target_luma=0.48)

    assert gamma < 1.0  # 감마 < 1 이 밝게 만든다
    assert to_gray(brightened).mean() > to_gray(dark).mean()


def test_adaptive_gamma_darkens_bright_frame() -> None:
    bright = np.clip(_textured().astype(np.float32) * 2.2, 0, 255).astype(np.uint8)
    darkened, gamma = adaptive_gamma(bright, target_luma=0.48)

    assert gamma > 1.0
    assert to_gray(darkened).mean() < to_gray(bright).mean()


def test_adaptive_gamma_is_noop_when_already_on_target() -> None:
    image = _textured()
    mean = float(to_gray(image).mean()) / 255.0
    _, gamma = adaptive_gamma(image, target_luma=mean)
    assert gamma == 1.0


def test_adaptive_gamma_handles_all_black() -> None:
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    result, gamma = adaptive_gamma(black)
    assert gamma == 1.0
    assert np.array_equal(result, black)


def test_clahe_raises_local_contrast() -> None:
    low_contrast = np.clip(_textured().astype(np.float32) * 0.25 + 96, 0, 255).astype(np.uint8)
    boosted = clahe_luminance(low_contrast)
    assert compute_metrics(boosted).rms_contrast > compute_metrics(low_contrast).rms_contrast


def test_unsharp_increases_sharpness() -> None:
    blurred = _blur(_textured(), 2.0)
    sharpened = unsharp_mask(blurred, amount=0.8)
    assert compute_metrics(sharpened).sharpness_lapvar > compute_metrics(blurred).sharpness_lapvar


def test_unsharp_zero_amount_is_noop() -> None:
    image = _textured()
    assert np.array_equal(unsharp_mask(image, amount=0.0), image)


def test_rescue_exposure_lifts_dark_frame() -> None:
    dark = (_blue_cast(_textured()).astype(np.float32) * 0.18).astype(np.uint8)
    rescued, report = rescue_exposure(dark)

    before = compute_metrics(dark)
    after = compute_metrics(rescued)
    assert after.luma_mean > before.luma_mean
    assert after.rms_contrast > before.rms_contrast
    assert not report.denoised  # 구제 경로는 디노이징을 하지 않는다
    assert "clahe" in report.stages


def test_rescue_exposure_does_not_mutate_input() -> None:
    dark = (_textured().astype(np.float32) * 0.2).astype(np.uint8)
    original = dark.copy()
    rescue_exposure(dark)
    assert np.array_equal(dark, original)


def test_enhance_denoises_boosted_noisy_frame() -> None:
    """저조도 + 노이즈 프레임은 감마 후 sigma 기준으로 디노이징이 걸려야 한다.

    원본 sigma로 판정하면 어두운 화소의 노이즈 진폭이 눌려 있어 임계를 넘지 못한다.
    이 테스트가 그 회귀를 막는다.
    """
    rng = np.random.default_rng(3)
    # 센서 노이즈는 신호와 함께 눌린다. 노이즈를 더한 뒤 어둡게 만드는 순서가 실제에 가깝다.
    scene = _textured().astype(np.float32)
    noisy_scene = scene + rng.normal(0.0, 16.0, scene.shape)
    dark_noisy = np.clip(noisy_scene * 0.15, 0, 255).astype(np.uint8)

    raw_sigma = noise_sigma(to_gray(dark_noisy))
    # 원본 기준 sigma는 임계 아래 → 원본만 보고 판정하면 디노이징이 걸리지 않는다.
    assert raw_sigma < EnhanceConfig().denoise_sigma_threshold

    _, report = enhance_for_vlm(
        dark_noisy, noise_sigma=raw_sigma, sharpness_lapvar=10.0, sharpness_ref=1000.0
    )
    assert report.denoised
    assert any(stage.startswith("denoise") for stage in report.stages)


def test_enhance_skips_denoise_on_clean_frame() -> None:
    clean = _textured()
    metrics = compute_metrics(clean)
    _, report = enhance_for_vlm(
        clean,
        noise_sigma=metrics.noise_sigma,
        sharpness_lapvar=metrics.sharpness_lapvar,
        sharpness_ref=metrics.sharpness_lapvar,
    )
    assert not report.denoised
    assert not report.sharpened  # 이미 기준 선명도라 샤프닝도 걸리지 않는다


def test_enhance_sharpens_only_below_reference() -> None:
    image = _textured()
    metrics = compute_metrics(image)

    _, low = enhance_for_vlm(
        image,
        noise_sigma=metrics.noise_sigma,
        sharpness_lapvar=metrics.sharpness_lapvar,
        sharpness_ref=metrics.sharpness_lapvar * 10.0,
    )
    assert low.sharpened


def test_enhance_report_records_stage_order() -> None:
    dark = (_blue_cast(_textured()).astype(np.float32) * 0.2).astype(np.uint8)
    metrics = compute_metrics(dark)
    _, report = enhance_for_vlm(
        dark,
        noise_sigma=metrics.noise_sigma,
        sharpness_lapvar=metrics.sharpness_lapvar,
        sharpness_ref=metrics.sharpness_lapvar,
    )
    stages = list(report.stages)
    assert stages.index("white_balance") < stages.index("gamma") < stages.index("clahe")


@pytest.mark.parametrize("scale", [0.1, 0.3, 1.0, 1.8])
def test_enhance_output_shape_and_dtype_preserved(scale: float) -> None:
    image = np.clip(_textured().astype(np.float32) * scale, 0, 255).astype(np.uint8)
    metrics = compute_metrics(image)
    result, _ = enhance_for_vlm(
        image,
        noise_sigma=metrics.noise_sigma,
        sharpness_lapvar=metrics.sharpness_lapvar,
        sharpness_ref=max(metrics.sharpness_lapvar, 1.0),
    )
    assert result.shape == image.shape
    assert result.dtype == np.uint8
