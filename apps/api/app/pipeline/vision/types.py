"""비전 프론트엔드 공용 타입.

파이프라인 규칙에 따라 모든 입출력은 Pydantic 모델이며 DB 의존이 없다.
프레임 픽셀 자체(`numpy` 배열)는 직렬화하지 않고 `FrameRef.path`로만 참조한다.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# BGR uint8 프레임. 런타임 타입은 numpy.ndarray 이지만 Pydantic 모델에는 넣지 않는다.
ColorSpace = Literal["bgr"]


class RejectReason(StrEnum):
    """프레임이 VLM 입력에서 탈락한 이유."""

    BLUR = "blur"
    UNDEREXPOSED = "underexposed"
    OVEREXPOSED = "overexposed"
    DUPLICATE = "duplicate"
    BUDGET = "budget"


class QualityMetrics(BaseModel):
    """단일 프레임의 무참조(no-reference) 화질 지표.

    모두 원본 프레임(보정 전) 기준으로 계산한다. 정규화 값이 아닌 raw 값을 함께 보관해
    영상별로 상대 임계값(퍼센타일)을 쓸 수 있도록 한다.
    """

    model_config = ConfigDict(frozen=True)

    sharpness_lapvar: float = Field(description="Laplacian 응답의 분산. 클수록 선명")
    sharpness_tenengrad: float = Field(description="Sobel 그래디언트 에너지 평균")
    luma_mean: float = Field(ge=0.0, le=1.0, description="평균 휘도(0~1)")
    clipped_low_ratio: float = Field(ge=0.0, le=1.0, description="흑점 포화 픽셀 비율")
    clipped_high_ratio: float = Field(ge=0.0, le=1.0, description="백점 포화 픽셀 비율")
    rms_contrast: float = Field(ge=0.0, description="휘도 표준편차(0~1 스케일)")
    noise_sigma: float = Field(ge=0.0, description="Immerkaer 추정 노이즈 표준편차(0~255 스케일)")
    colorfulness: float = Field(ge=0.0, description="Hasler-Susstrunk colorfulness")
    textness: float = Field(ge=0.0, le=1.0, description="간판/문자 영역 추정 점유율 기반 점수")


class FrameStats(BaseModel):
    """샘플링된 프레임 1장의 메타데이터."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="샘플링 순서(0-based)")
    timestamp_sec: float = Field(ge=0.0, description="영상 내 시각(초)")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    metrics: QualityMetrics = Field(description="원본(보정 전) 지표")
    rescued: bool = Field(
        default=False, description="화질 게이트 전에 저조도 구제 보정이 적용되었는지"
    )
    metrics_rescued: QualityMetrics | None = Field(
        default=None, description="구제 보정 후 지표. rescued=False면 None"
    )
    text_boxes: int | None = Field(
        default=None,
        ge=0,
        description="학습된 문자 검출기의 박스 수. selection_strategy='text_nms' 에서 게이트를"
        " 통과한 프레임에만 채운다",
    )

    @property
    def effective_metrics(self) -> QualityMetrics:
        """게이트·점수 계산에 쓰는 지표. 구제된 프레임은 보정 후 값을 쓴다."""
        return self.metrics_rescued or self.metrics


class ShotSegment(BaseModel):
    """임베딩 유사도로 묶인 연속 프레임 구간(샷)."""

    model_config = ConfigDict(frozen=True)

    shot_id: int = Field(ge=0)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    frame_indices: list[int] = Field(min_length=1)


class Keyframe(BaseModel):
    """VLM에 실제로 올라가는 최종 프레임."""

    model_config = ConfigDict(frozen=True)

    frame_index: int = Field(ge=0)
    shot_id: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    path: Path = Field(description="저장된 JPEG 경로(보정 적용 후)")
    selection_score: float = Field(description="샷 내 대표 선정에 쓰인 종합 점수")
    enhanced: bool = Field(description="미니 ISP 보정이 적용되었는지")
    metrics: QualityMetrics = Field(description="보정 전 원본 지표")
    metrics_after: QualityMetrics | None = Field(
        default=None, description="보정 후 지표. enhanced=False면 None"
    )


class RejectedFrame(BaseModel):
    """탈락 프레임 기록. 감축률 산출과 디버깅에 사용."""

    model_config = ConfigDict(frozen=True)

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    reason: RejectReason
    detail: str = ""


class VisionFrontendConfig(BaseModel):
    """키프레임 선별 동작을 결정하는 설정.

    기본값은 5~15분 길이의 여행 브이로그(대부분 핸드헬드, 실내외 혼합)를 기준으로 잡았다.
    """

    model_config = ConfigDict(frozen=True)

    # --- 디코딩 ---
    sample_fps: float = Field(default=1.0, gt=0.0, description="초당 샘플링할 프레임 수")
    max_decoded_frames: int = Field(default=900, gt=0, description="디코딩 상한(긴 영상 보호)")
    long_edge_px: int = Field(default=1280, gt=0, description="디코딩 후 긴 변 리사이즈 목표")

    # --- 화질 게이트 ---
    blur_neighbor_ratio: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="시간상 이웃 프레임 선명도 중앙값의 이 비율 아래면 흔들린 프레임으로 본다."
        " 0이면 상대 컷 비활성. 영상 전체 분포의 하위 N%를 자르는 방식은 '흔들린 프레임이"
        " 항상 N% 존재한다'고 가정하는 셈이어서, 텍스처가 적은 장면을 통째로 버린다",
    )
    blur_neighbor_window_sec: float = Field(
        default=3.0, gt=0.0, description="이웃 판단에 쓰는 시간 창(±초)"
    )
    blur_global_floor_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="영상 전체 선명도 중앙값의 이 비율 아래면 이웃과 무관하게 탈락."
        " 이웃 기준만 쓰면 수 초간 이어지는 흔들림 구간은 '이웃도 똑같이 흐리므로' 통과한다."
        " 영상 자신의 스케일로 잡은 절대 하한이 그걸 막는다",
    )
    min_sharpness_lapvar: float = Field(
        default=8.0, ge=0.0, description="절대 하한. 영상 전체가 흐릴 때 과도한 컷 방지용 하한선"
    )
    clipped_ratio_max: float = Field(
        default=0.30, ge=0.0, le=1.0, description="흑/백 포화 비율 상한(구제 후 기준)"
    )
    luma_range: tuple[float, float] = Field(
        default=(0.06, 0.94), description="허용 평균 휘도 범위(구제 후 기준)"
    )

    # --- 저조도 구제 (화질 게이트보다 먼저 실행) ---
    rescue_luma_below: float = Field(
        default=0.18,
        ge=0.0,
        le=1.0,
        description="평균 휘도가 이 값 아래면 게이트 전에 WB+감마+CLAHE로 구제 시도",
    )
    rescue_clip_low_above: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
        description="흑점 포화 비율이 이 값 이상이면 구제 시도",
    )
    unrecoverable_luma_below: float = Field(
        default=0.015,
        ge=0.0,
        le=1.0,
        description="이보다 어두우면 신호가 남아 있지 않다고 보고 구제 없이 탈락",
    )
    unrecoverable_clip_low_above: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="흑점에 이 비율 이상 몰리면 복원 불가로 탈락",
    )

    # --- 중복 제거 ---
    scene_similarity_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="이 값 이상 유사하면 같은 샷으로 묶음. None이면 임베더 권장값"
        " (perceptual 0.88 / dinov3 0.97) — 유사도 스케일이 임베더마다 다르다",
    )
    duplicate_shot_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="이 값 이상 유사한 떨어진 샷은 재방문 중복으로 제거. None이면 임베더 권장값"
        " (perceptual 0.95 / dinov3 0.97)",
    )
    min_shot_gap_sec: float = Field(
        default=1.5, ge=0.0, description="샷 경계 최소 간격. 손떨림에 의한 과분할 방지"
    )
    max_shot_cut_rate: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="샷 경계로 삼을 인접쌍의 최대 비율. `scene_similarity_threshold`를"
        " 지정하지 않았을 때만 쓰이며, 이 비율을 넘지 않도록 임계를 영상별로 낮춘다."
        " 고정 임계는 패닝이 많은 영상에서 장면을 과분할한다",
    )

    # --- 선별 ---
    max_keyframes: int = Field(default=16, gt=0, description="VLM에 올릴 최대 프레임 수")
    selection_strategy: Literal["diverse", "text_nms"] = Field(
        default="diverse",
        description="'diverse' = 샷 분할 → 샷별 대표 → 중복 샷 제거 → 시각 다양성(k-center)"
        " 예산 컷."
        " 'text_nms' = 게이트 통과 프레임을 학습된 문자 검출기(PP-OCRv4 DB) 박스 수 순으로 보며"
        " 시간 간격을 두고 고른다. 유튜브 개발 세트에서 diverse 는 무작위 수준이었다"
        " (docs/vision-frontend.md). text_nms 는 사전 등록 테스트에서 기각됐다"
        " (35곳 중 4 vs diverse 3, 무작위 기댓값 3.9) — 실험 재현용으로만 남긴다",
    )
    text_nms_gap_ratio: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="text_nms 의 최소 시간 간격 = 이 비율 × (샘플 수 / max_keyframes)."
        " 사전 등록 테스트에 쓴 값 (docs/vision-frontend-decisions.md #13–15)",
    )
    text_weight: float = Field(
        default=0.35, ge=0.0, le=1.0, description="대표 프레임 점수에서 간판 텍스트 가중치"
    )

    # --- 보정 ---
    enhance: bool = Field(default=True, description="미니 ISP 보정 적용 여부")
    enhance_textness_drop_limit: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="보정 후 문자 영역 점수가 보정 전보다 이 값 이상 떨어지면 보정을 버리고"
        " 값싼 구제본(또는 원본)을 내보낸다. 저조도에서 디노이징이 간판 획을 지우는 경우를"
        " 막는 안전장치",
    )
    jpeg_quality: int = Field(default=92, ge=1, le=100)


class SelectionSummary(BaseModel):
    """벤치마크/로그용 집계."""

    model_config = ConfigDict(frozen=True)

    video_duration_sec: float = Field(ge=0.0)
    decoded_frames: int = Field(ge=0)
    rescued_frames: int = Field(
        default=0, ge=0, description="게이트 전에 저조도 구제 보정을 받은 프레임 수"
    )
    after_quality_gate: int = Field(ge=0)
    shots_detected: int = Field(ge=0)
    selected_frames: int = Field(ge=0)
    reduction_ratio: float = Field(
        ge=0.0, le=1.0, description="1 - selected/decoded. 클수록 많이 줄인 것"
    )
    embedder: str = Field(description="사용된 임베더 이름")
    elapsed_sec: float = Field(ge=0.0)


class KeyframeSelection(BaseModel):
    """`select_keyframes`의 반환값."""

    keyframes: list[Keyframe]
    shots: list[ShotSegment]
    rejected: list[RejectedFrame]
    frame_stats: list[FrameStats]
    summary: SelectionSummary
    config: VisionFrontendConfig
