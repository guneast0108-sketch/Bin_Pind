"""무참조(no-reference) 화질 지표.

VLM에 올릴 프레임을 고르는 데 필요한 것은 "이 프레임이 읽을 만한가"이지 절대 화질 점수가
아니다. 그래서 학습 기반 IQA 대신 계산이 싸고 해석 가능한 고전 지표를 쓴다. 각 지표의 출처:

- Laplacian variance      : Pech-Pacheco et al., ICPR 2000 (자동초점 지표 비교)
- Tenengrad (Sobel energy): Krotkov, IJCV 1987
- Noise sigma             : Immerkaer, "Fast Noise Variance Estimation", CVIU 1996
- Colorfulness            : Hasler & Susstrunk, SPIE 2003

`textness`는 `textness.py`에서 계산해 주입한다(MSER 기반, 여기서 계산하지 않음).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.pipeline.vision.types import QualityMetrics

BgrImage = NDArray[np.uint8]
GrayImage = NDArray[np.uint8]

# Immerkaer(1996)의 라플라시안 차분 커널. 두 라플라시안의 차로 신호를 지우고 노이즈만 남긴다.
_IMMERKAER_KERNEL: NDArray[np.float32] = np.array(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32
)


def to_gray(image: BgrImage) -> GrayImage:
    """BGR → 8비트 그레이스케일."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)


def sharpness_lapvar(gray: GrayImage) -> float:
    """Laplacian 응답의 분산. 초점이 맞을수록 고주파 에너지가 커진다."""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float(lap.var())


def sharpness_tenengrad(gray: GrayImage) -> float:
    """Sobel 그래디언트 제곱합의 평균(Tenengrad)."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def noise_sigma(gray: GrayImage) -> float:
    """Immerkaer(1996) 빠른 노이즈 표준편차 추정.

    sigma = sqrt(pi/2) * (1 / (6 * (W-2) * (H-2))) * sum |I * M|
    """
    height, width = gray.shape[:2]
    if height < 3 or width < 3:
        return 0.0
    response = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, _IMMERKAER_KERNEL)
    inner = response[1:-1, 1:-1]
    denom = 6.0 * (width - 2) * (height - 2)
    return float(np.sqrt(np.pi / 2.0) * np.abs(inner).sum() / denom)


def exposure_stats(gray: GrayImage) -> tuple[float, float, float, float]:
    """(평균 휘도, 흑점 포화 비율, 백점 포화 비율, RMS 콘트라스트). 모두 0~1 스케일."""
    luma = gray.astype(np.float32) / 255.0
    total = float(luma.size)
    low = float(np.count_nonzero(gray <= 8)) / total
    high = float(np.count_nonzero(gray >= 247)) / total
    return float(luma.mean()), low, high, float(luma.std())


def colorfulness(image: BgrImage) -> float:
    """Hasler & Susstrunk(2003) colorfulness 지표.

    간판·메뉴판·인테리어처럼 색이 있는 장면과 무채색 벽/하늘을 구분하는 데 쓴다.
    """
    blue, green, red = (channel.astype(np.float32) for channel in cv2.split(image))
    rg = red - green
    yb = 0.5 * (red + green) - blue
    std_root = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
    mean_root = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return std_root + 0.3 * mean_root


def compute_metrics(image: BgrImage, *, textness: float = 0.0) -> QualityMetrics:
    """프레임 1장의 전체 지표를 계산한다.

    Args:
        image: BGR uint8 프레임.
        textness: `textness.py`에서 계산한 문자 영역 점수(0~1).
    """
    gray = to_gray(image)
    luma_mean, low, high, rms = exposure_stats(gray)
    return QualityMetrics(
        sharpness_lapvar=sharpness_lapvar(gray),
        sharpness_tenengrad=sharpness_tenengrad(gray),
        luma_mean=luma_mean,
        clipped_low_ratio=low,
        clipped_high_ratio=high,
        rms_contrast=rms,
        noise_sigma=noise_sigma(gray),
        colorfulness=colorfulness(image),
        textness=float(np.clip(textness, 0.0, 1.0)),
    )


def readability_score(
    metrics: QualityMetrics,
    *,
    sharpness_ref: float,
    text_weight: float = 0.35,
) -> float:
    """ "VLM이 이 프레임에서 장소를 읽어낼 가능성"에 대한 종합 점수(0~1 근방).

    선명도는 영상 내 상위값(`sharpness_ref`)으로 정규화한다. 절대 임계값은 촬영 기기와
    비트레이트에 따라 크게 달라지므로 영상별 상대 기준이 더 안정적이다.

    Args:
        metrics: 해당 프레임 지표.
        sharpness_ref: 이 영상의 선명도 기준값(보통 90퍼센타일).
        text_weight: 간판/문자 점수에 줄 가중치.
    """
    sharp = float(np.clip(metrics.sharpness_lapvar / max(sharpness_ref, 1e-6), 0.0, 1.0))

    # 중간 노출에서 1, 양 극단에서 0으로 떨어지는 삼각형 가중.
    exposure = 1.0 - abs(metrics.luma_mean - 0.5) * 2.0
    exposure = float(np.clip(exposure, 0.0, 1.0))
    clipping = metrics.clipped_low_ratio + metrics.clipped_high_ratio
    clipping_penalty = float(np.clip(clipping, 0.0, 1.0))

    contrast = float(np.clip(metrics.rms_contrast / 0.25, 0.0, 1.0))
    noise_penalty = float(np.clip(metrics.noise_sigma / 12.0, 0.0, 1.0))
    color = float(np.clip(metrics.colorfulness / 60.0, 0.0, 1.0))

    base = (
        0.40 * sharp
        + 0.20 * exposure
        + 0.15 * contrast
        + 0.10 * color
        - 0.15 * clipping_penalty
        - 0.10 * noise_penalty
    )
    score = (1.0 - text_weight) * base + text_weight * metrics.textness
    return float(np.clip(score, 0.0, 1.0))
