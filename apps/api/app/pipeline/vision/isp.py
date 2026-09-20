"""미니 ISP — VLM 입력 전처리.

브이로그 프레임은 이미 카메라 ISP를 거친 sRGB JPEG다. 그래도 실내 저조도, 혼합 색온도
(백열등 + 창 채광), 노출 언더는 흔하고, 이런 프레임에서 간판 글자가 뭉개져 VLM이 장소를
놓친다. 여기서는 sRGB 도메인에서 ISP 후단 블록만 재현해 "읽을 수 있는" 프레임으로 만든다.

적용 순서와 근거:

1. 화이트밸런스 — Shades-of-Gray (Finlayson & Trezzi, CIC 2004).
   Minkowski norm p 하나로 Gray-World(p=1)와 White-Patch(p=inf)를 잇는 계열. 간판 색이
   광원색에 끌려가면 텍스트 대비가 떨어지므로 색 항상성을 먼저 잡는다.
2. 톤 — 목표 평균 휘도에 맞추는 적응 감마.
   `gamma = log(target) / log(mean)` 로 한 번에 계산한다(전역 톤커브 1회, 히스토그램 평활화와
   달리 색상 왜곡이 없음).
3. 노이즈 — 감마로 들어올린 뒤 조건부 디노이징.
   저조도 프레임을 밝히면 섀도우 노이즈가 같이 증폭된다. 중요한 건 판정 시점이다. 원본
   어두운 프레임의 sigma는 값이 눌려 있어 작게 나오므로, 감마 *후* 이미지에서 다시 추정해
   판정한다(원본 기준으로 판정하면 정작 필요한 저조도 프레임에서 디노이징이 걸리지 않는다).
   임계를 넘을 때만 적용해 평상시 비용은 0이다.
4. 로컬 콘트라스트 — LAB의 L 채널에만 CLAHE.
   색 채널을 건드리지 않아 채도 과장 없이 간판 글자 경계를 살린다.
5. 샤프닝 — 언샤프 마스킹, 선명도가 기준 이하일 때만.
   이미 선명한 프레임에 걸면 링잉과 노이즈만 늘어난다.

모두 순수 함수이며 입력 배열을 변경하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from app.pipeline.vision.quality import noise_sigma as estimate_noise_sigma
from app.pipeline.vision.quality import to_gray

BgrImage = NDArray[np.uint8]


@dataclass(frozen=True)
class EnhanceConfig:
    """미니 ISP 파라미터."""

    minkowski_p: float = 6.0
    """Shades-of-Gray 차수. 1=Gray-World, 큰 값=White-Patch 쪽."""

    wb_max_gain: float = 2.5
    """채널 이득 상한. 단색 프레임에서 색이 뒤집히는 것을 막는다."""

    target_luma: float = 0.48
    """감마 보정의 목표 평균 휘도."""

    gamma_limits: tuple[float, float] = (0.45, 2.2)
    """감마 허용 범위. 과보정 방지."""

    denoise_sigma_threshold: float = 2.5
    """감마 *후* 추정 노이즈(0~255 스케일)가 이 값을 넘을 때만 디노이징.

    원본 기준이 아니라 감마 후 기준이라는 점이 중요하다. 잘 노출된 프레임은 감마를 거쳐도
    sigma가 1~2 수준이라 걸리지 않고, 저조도에서 들어올린 프레임만 선택적으로 걸린다.
    """

    denoise_strength_scale: float = 2.5
    """fastNlMeans 의 h 를 추정 sigma의 몇 배로 잡을지. h ~= scale * sigma."""

    denoise_strength_limits: tuple[int, int] = (3, 12)
    """h 의 하한/상한. 너무 크면 간판 글자 획까지 지워진다."""

    clahe_clip: float = 2.0
    clahe_grid: int = 8

    unsharp_sigma: float = 1.2
    unsharp_amount: float = 0.6
    """언샤프 마스킹 강도. 0이면 비활성."""


@dataclass(frozen=True)
class EnhanceReport:
    """어떤 단계가 실제로 적용되었는지의 기록. 벤치마크 표에 그대로 들어간다."""

    wb_gains: tuple[float, float, float]
    gamma: float
    denoised: bool
    sharpened: bool
    stages: tuple[str, ...] = field(default=())


def white_balance_shades_of_gray(
    image: BgrImage, *, minkowski_p: float = 6.0, max_gain: float = 2.5
) -> tuple[BgrImage, tuple[float, float, float]]:
    """Shades-of-Gray 화이트밸런스.

    각 채널의 Minkowski p-노름을 구해 전체 평균에 맞추는 이득을 적용한다.

    Returns:
        (보정 이미지, (B, G, R) 이득)
    """
    work = image.astype(np.float32)
    norms: list[float] = []
    for channel in range(3):
        values = work[:, :, channel]
        norm = float(np.power(np.mean(np.power(values, minkowski_p)), 1.0 / minkowski_p))
        norms.append(max(norm, 1e-6))

    target = float(np.mean(norms))
    gains = tuple(float(np.clip(target / norm, 1.0 / max_gain, max_gain)) for norm in norms)

    for channel in range(3):
        work[:, :, channel] *= gains[channel]

    balanced = np.clip(work, 0.0, 255.0).astype(np.uint8)
    return balanced, (gains[0], gains[1], gains[2])


def adaptive_gamma(
    image: BgrImage,
    *,
    target_luma: float = 0.48,
    limits: tuple[float, float] = (0.45, 2.2),
) -> tuple[BgrImage, float]:
    """평균 휘도를 목표값으로 끌어오는 전역 감마 보정.

    Returns:
        (보정 이미지, 적용된 감마). 감마 1.0이면 보정하지 않은 것.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean = float(gray.mean())
    if mean <= 1e-4 or mean >= 1.0 - 1e-4:
        return image.copy(), 1.0

    gamma = float(np.log(target_luma) / np.log(mean))
    gamma = float(np.clip(gamma, limits[0], limits[1]))
    if abs(gamma - 1.0) < 0.02:
        return image.copy(), 1.0

    table = np.clip(np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0, 0, 255)
    lut = table.astype(np.uint8)
    return cv2.LUT(image, lut).astype(np.uint8), gamma


def clahe_luminance(image: BgrImage, *, clip: float = 2.0, grid: int = 8) -> BgrImage:
    """LAB의 L 채널에만 CLAHE를 적용한다(색 왜곡 없는 로컬 콘트라스트 강화)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    operator = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    merged = cv2.merge((operator.apply(lightness), a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR).astype(np.uint8)


def unsharp_mask(image: BgrImage, *, sigma: float = 1.2, amount: float = 0.6) -> BgrImage:
    """언샤프 마스킹: out = in + amount * (in - blur(in))."""
    if amount <= 0.0:
        return image.copy()
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharp = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0.0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def rescue_exposure(
    image: BgrImage, *, config: EnhanceConfig | None = None
) -> tuple[BgrImage, EnhanceReport]:
    """저조도 구제 전용 경로 — 화이트밸런스 + 감마 + CLAHE만.

    `enhance_for_vlm`과 달리 디노이징과 샤프닝을 하지 않아 프레임당 수 ms로 끝난다.
    화질 게이트 *앞*에서 모든 어두운 프레임에 적용하는 것이 목적이기 때문이다.

    이 순서가 중요하다. 노출이 무너진 프레임을 먼저 버리고 나중에 보정하면, 실내 저조도에서
    찍힌 간판(미술관·전시장·술집)은 영상에 분명히 있는데도 파이프라인에서 사라진다.
    구제 가능한 프레임은 먼저 살려 놓고, 그 다음에 "살려도 못 읽는" 프레임만 버린다.
    """
    cfg = config or EnhanceConfig()
    stages: list[str] = []

    result, gains = white_balance_shades_of_gray(
        image, minkowski_p=cfg.minkowski_p, max_gain=cfg.wb_max_gain
    )
    if any(abs(gain - 1.0) > 0.02 for gain in gains):
        stages.append("white_balance")

    result, gamma = adaptive_gamma(result, target_luma=cfg.target_luma, limits=cfg.gamma_limits)
    if gamma != 1.0:
        stages.append("gamma")

    result = clahe_luminance(result, clip=cfg.clahe_clip, grid=cfg.clahe_grid)
    stages.append("clahe")

    report = EnhanceReport(
        wb_gains=gains, gamma=gamma, denoised=False, sharpened=False, stages=tuple(stages)
    )
    return result, report


def enhance_for_vlm(
    image: BgrImage,
    *,
    noise_sigma: float,
    sharpness_lapvar: float,
    sharpness_ref: float,
    config: EnhanceConfig | None = None,
) -> tuple[BgrImage, EnhanceReport]:
    """미니 ISP 전체를 적용한다.

    Args:
        image: BGR uint8 프레임.
        noise_sigma: 원본 프레임의 `quality.noise_sigma` 추정값. 감마 후 재추정값과
            비교해 큰 쪽으로 디노이징 여부를 결정한다.
        sharpness_lapvar: 이 프레임의 Laplacian 분산.
        sharpness_ref: 영상 기준 선명도(보통 90퍼센타일). 샤프닝 여부를 결정한다.
        config: 파라미터. None이면 기본값.

    Returns:
        (보정 이미지, 적용 리포트)
    """
    cfg = config or EnhanceConfig()
    stages: list[str] = []

    result, gains = white_balance_shades_of_gray(
        image, minkowski_p=cfg.minkowski_p, max_gain=cfg.wb_max_gain
    )
    if any(abs(gain - 1.0) > 0.02 for gain in gains):
        stages.append("white_balance")

    result, gamma = adaptive_gamma(result, target_luma=cfg.target_luma, limits=cfg.gamma_limits)
    if gamma != 1.0:
        stages.append("gamma")

    # 감마로 들어올린 뒤의 노이즈로 판정한다. 원본 sigma는 저조도에서 과소평가된다
    # (어두운 화소는 값 자체가 눌려 있어 노이즈 진폭도 같이 눌린다).
    # 뒤따르는 CLAHE가 로컬 노이즈를 한 번 더 키우므로, 디노이징은 CLAHE 앞에 두고
    # 강도를 감마 후 sigma에 비례시킨다.
    post_tone_sigma = max(noise_sigma, estimate_noise_sigma(to_gray(result)))
    denoised = post_tone_sigma > cfg.denoise_sigma_threshold
    if denoised:
        low, high = cfg.denoise_strength_limits
        strength = int(np.clip(round(cfg.denoise_strength_scale * post_tone_sigma), low, high))
        result = cv2.fastNlMeansDenoisingColored(result, None, strength, strength, 7, 21).astype(
            np.uint8
        )
        stages.append(f"denoise(h={strength})")

    result = clahe_luminance(result, clip=cfg.clahe_clip, grid=cfg.clahe_grid)
    stages.append("clahe")

    sharpened = sharpness_lapvar < 0.6 * max(sharpness_ref, 1e-6)
    if sharpened:
        result = unsharp_mask(result, sigma=cfg.unsharp_sigma, amount=cfg.unsharp_amount)
        stages.append("unsharp")

    report = EnhanceReport(
        wb_gains=gains,
        gamma=gamma,
        denoised=denoised,
        sharpened=sharpened,
        stages=tuple(stages),
    )
    return result, report
