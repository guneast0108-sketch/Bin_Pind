"""비전 프론트엔드 — VLM 호출 앞단의 프레임 선별·보정 단계.

`analyze_vision`(Gemini Vision, B안)에 프레임을 그대로 넘기는 대신, 이 패키지가 먼저
"읽을 수 있고 서로 다른" 프레임만 골라낸다. 목적은 두 가지다.

1. 비용 — 1 fps로 뽑은 프레임의 대부분은 앞 프레임과 거의 같다. 샷 단위로 대표 1장만
   남기면 VLM 입력 프레임 수가 한 자리 수 퍼센트대로 줄고, 그만큼 토큰 비용도 줄어든다.
2. 정확도 — 흔들리거나 노출이 무너진 프레임은 간판을 읽을 수 없다. 이런 프레임을 걸러내고,
   남은 프레임은 화이트밸런스·톤·로컬 콘트라스트를 보정해 글자 대비를 살려 넘긴다.

사용 예:

    from pathlib import Path
    from app.pipeline.vision import VisionFrontendConfig, build_embedder, select_keyframes

    selection = select_keyframes(
        Path("video.mp4"),
        Path("/tmp/pind-work"),
        config=VisionFrontendConfig(sample_fps=1.0, max_keyframes=16),
        embedder=build_embedder("auto"),
    )
    frame_paths = [keyframe.path for keyframe in selection.keyframes]
"""

from app.pipeline.vision.embed import (
    DEFAULT_DINOV3_MODEL,
    Dinov3Embedder,
    Embedder,
    PerceptualEmbedder,
    build_embedder,
)
from app.pipeline.vision.isp import (
    EnhanceConfig,
    EnhanceReport,
    enhance_for_vlm,
    rescue_exposure,
)
from app.pipeline.vision.quality import compute_metrics, readability_score
from app.pipeline.vision.select import select_keyframes
from app.pipeline.vision.textdet import (
    DbTextDetector,
    TextDetection,
    TextDetector,
    TextDetectorUnavailable,
)
from app.pipeline.vision.textness import TextnessResult, textness
from app.pipeline.vision.types import (
    FrameStats,
    Keyframe,
    KeyframeSelection,
    QualityMetrics,
    RejectedFrame,
    RejectReason,
    SelectionSummary,
    ShotSegment,
    VisionFrontendConfig,
)

__all__ = [
    "DEFAULT_DINOV3_MODEL",
    "DbTextDetector",
    "Dinov3Embedder",
    "Embedder",
    "EnhanceConfig",
    "EnhanceReport",
    "FrameStats",
    "Keyframe",
    "KeyframeSelection",
    "PerceptualEmbedder",
    "QualityMetrics",
    "RejectReason",
    "RejectedFrame",
    "SelectionSummary",
    "ShotSegment",
    "TextDetection",
    "TextDetector",
    "TextDetectorUnavailable",
    "TextnessResult",
    "VisionFrontendConfig",
    "build_embedder",
    "compute_metrics",
    "enhance_for_vlm",
    "readability_score",
    "rescue_exposure",
    "select_keyframes",
    "textness",
]
