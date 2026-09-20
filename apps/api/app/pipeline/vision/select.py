"""키프레임 선별 오케스트레이터.

`select_keyframes`가 이 패키지의 유일한 공개 진입점이다. 전체 흐름:

    디코딩(ffmpeg, 균일 샘플링)
      → 프레임별 화질 지표 + 간판 텍스트 점수 + 저조도 구제 + 임베딩  [청크 스트리밍]
      → 화질 게이트 (복원해도 못 읽는 프레임만 탈락)
      → [selection_strategy="diverse"]
          샷 분할 (인접 임베딩 유사도)
          → 샷별 대표 1장 (readability_score 최대)
          → 전역 중복 샷 제거 (같은 가게 재방문 컷 병합)
          → 예산 컷 (시각 다양성, max_keyframes)
        [selection_strategy="text_nms"]
          게이트 통과 프레임에 학습된 문자 검출기 → 박스 수 순 + 시간 간격 (max_keyframes)
      → 미니 ISP 보정 후 JPEG 저장

저조도 구제가 화질 게이트보다 **앞에** 있는 것이 의도된 순서다. 게이트를 먼저 두면 실내에서
찍힌 어두운 프레임이 전부 언더노출로 탈락해, 영상에 분명히 나온 실내 장소가 결과에서 통째로
사라진다. 그래서 게이트의 역할은 "품질이 낮은 프레임 제거"가 아니라 "복원해도 읽을 수 없는
프레임 제거"다.

메모리 상한을 위해 프레임 픽셀은 청크(기본 32장)만 들고 있고, 최종 선별된 프레임만 다시 읽는다.
파이프라인 규칙대로 DB에 접근하지 않으며, 임시 디렉토리는 호출자가 준 `work_dir` 아래에만 만든다.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import structlog
from numpy.typing import NDArray

from app.pipeline.vision import budget, decode, dedup, isp, quality, textness
from app.pipeline.vision.embed import Embedder, PerceptualEmbedder
from app.pipeline.vision.textdet import DbTextDetector, TextDetector
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

logger = structlog.get_logger(__name__)

_CHUNK_SIZE = 32


class _FrameAnalysis(NamedTuple):
    """프레임 1장의 분석 결과(픽셀은 포함하지 않음)."""

    metrics: QualityMetrics
    rescued: bool
    metrics_rescued: QualityMetrics | None


def _needs_rescue(metrics: QualityMetrics, config: VisionFrontendConfig) -> bool:
    """저조도 구제 대상인지. 복원 불가 수준이면 구제하지 않는다(그냥 탈락)."""
    if (
        metrics.luma_mean < config.unrecoverable_luma_below
        or metrics.clipped_low_ratio > config.unrecoverable_clip_low_above
    ):
        return False
    return (
        metrics.luma_mean < config.rescue_luma_below
        or metrics.clipped_low_ratio >= config.rescue_clip_low_above
    )


def _analyze_chunk(
    paths: list[Path], embedder: Embedder, config: VisionFrontendConfig
) -> tuple[list[_FrameAnalysis], NDArray[np.float32]]:
    """프레임 청크를 읽어 지표와 임베딩을 계산하고 픽셀은 버린다.

    어두운 프레임은 여기서 먼저 구제 보정을 받는다. 그래야 지표·문자 점수·임베딩이 모두
    "VLM이 실제로 보게 될 프레임" 기준으로 계산된다.
    """
    analyses: list[_FrameAnalysis] = []
    embed_inputs: list[decode.BgrImage] = []

    for path in paths:
        image = decode.read_bgr(path)
        metrics = quality.compute_metrics(image, textness=textness.textness(image).score)
        if _needs_rescue(metrics, config):
            rescued_image, _ = isp.rescue_exposure(image)
            metrics_rescued = quality.compute_metrics(
                rescued_image, textness=textness.textness(rescued_image).score
            )
            analyses.append(_FrameAnalysis(metrics, True, metrics_rescued))
            embed_inputs.append(rescued_image)
        else:
            analyses.append(_FrameAnalysis(metrics, False, None))
            embed_inputs.append(image)

    return analyses, embedder.embed(embed_inputs)


# 상대 blur 임계를 따로 계산하기 위한 그룹 최소 크기. 이보다 작은 그룹은 표본이
# 부족해 퍼센타일이 불안정하므로 전체 분포 기준으로 되돌린다.
_MIN_STRATUM_SIZE = 4


def _stratified_percentile(stats: list[FrameStats], percentile: float) -> dict[bool, float]:
    """구제된 프레임과 그렇지 않은 프레임의 선명도 퍼센타일을 따로 계산한다.

    표본이 `_MIN_STRATUM_SIZE` 미만인 그룹은 전체 분포 값으로 되돌린다.
    """
    everything = np.array(
        [item.effective_metrics.sharpness_lapvar for item in stats], dtype=np.float64
    )
    fallback = float(np.percentile(everything, percentile))

    values: dict[bool, float] = {}
    for rescued in (False, True):
        group = np.array(
            [item.effective_metrics.sharpness_lapvar for item in stats if item.rescued is rescued],
            dtype=np.float64,
        )
        values[rescued] = (
            float(np.percentile(group, percentile)) if group.size >= _MIN_STRATUM_SIZE else fallback
        )
    return values


# 이웃 기준 선명도를 계산할 때 필요한 최소 이웃 수. 이보다 적으면 구제 여부를 나누지 않는다.
_MIN_NEIGHBORS = 3


def keep_enhanced(
    *, candidate_textness: float, reference_textness: float, drop_limit: float
) -> bool:
    """전체 보정 결과를 쓸지, 값싼 구제본으로 되돌릴지 판단한다.

    보정은 대체로 글자 대비를 살리지만, 저조도에서는 디노이징 강도가 상한에 붙어
    간판 획까지 뭉개는 경우가 있다. 그런 프레임이야말로 잃으면 안 되는 프레임이라
    "보정이 문자 영역을 줄이면 보정을 쓰지 않는다"는 단방향 가드를 둔다.

    Args:
        candidate_textness: 전체 보정 후 문자 영역 점수.
        reference_textness: 보정 전(구제본 또는 원본) 문자 영역 점수.
        drop_limit: 허용 감소폭. 0이면 조금이라도 줄면 되돌린다.
    """
    return candidate_textness >= reference_textness - drop_limit


def _local_blur_cuts(stats: list[FrameStats], config: VisionFrontendConfig) -> list[float]:
    """프레임마다 **시간상 이웃**을 기준으로 선명도 하한을 계산한다.

    처음 구현은 영상 전체 선명도 분포의 하위 N%를 잘랐다. 이 방식에는 두 가지 결함이
    있고, 둘 다 실측에서 장소 손실로 나타났다(`benchmarks/validate_recall.py`).

    1. **선명도는 흔들림만의 함수가 아니다.** 벽면이 단조로운 가게는 흔들리지 않아도
       lapvar가 낮다. 전체 분포의 하위 25%를 자르면 "흔들린 프레임이 항상 25% 있다"고
       가정하는 셈이어서, 흔들림이 전혀 없는 영상에서도 프레임의 1/4을 버린다. 실측에서
       정상 노출 장소 한 곳이 이 경로로 **구간 전체가** 탈락했다(lapvar 13.1k, 컷 14.1k —
       사람 눈에는 완전히 선명한 프레임).
    2. **저조도 구제가 스케일을 바꾼다.** 구제는 CLAHE로 로컬 대비를 올리고 lapvar는
       대비의 제곱에 비례해 커진다(실측 약 1.8배). 구제 프레임과 정상 노출 프레임을 한
       분포에 섞으면 정상 노출 쪽이 하위권으로 밀린다.

    흔들림은 **시간적 아티팩트**다. 같은 장면 안에서 앞뒤 프레임보다 유독 흐린 프레임이
    흔들린 프레임이다. 그래서 기준을 전역 분포가 아니라 ±`blur_neighbor_window_sec`
    이웃의 중앙값으로 잡는다. 중앙값이라 창 안에 흔들린 프레임이 한두 장 섞여도 기준이
    끌려가지 않고, 장면 전체가 저텍스처면 기준도 같이 낮아져 아무것도 버리지 않는다.
    이웃은 구제 여부가 같은 프레임만 쓴다(1의 이유). 표본이 모자라면 창 전체로 되돌린다.

    Returns:
        `stats`와 같은 길이의 프레임별 lapvar 하한.
    """
    if not stats:
        return []

    times = np.array([item.timestamp_sec for item in stats], dtype=np.float64)
    lapvars = np.array(
        [item.effective_metrics.sharpness_lapvar for item in stats], dtype=np.float64
    )
    rescued = np.array([item.rescued for item in stats], dtype=bool)

    # 이웃 기준만 쓰면 수 초간 이어지는 흔들림 구간은 통과한다(이웃도 똑같이 흐리므로).
    # 영상 전체 중앙값에서 잡은 약한 절대 하한을 함께 둬서 그 경우를 막는다. 해상도·내용에
    # 따라 lapvar 스케일이 크게 달라지므로 고정값이 아니라 영상 자신의 스케일로 잡는다.
    global_floor = config.blur_global_floor_ratio * float(np.median(lapvars))

    cuts: list[float] = []
    for position in range(len(stats)):
        in_window = np.abs(times - times[position]) <= config.blur_neighbor_window_sec
        same_stratum = in_window & (rescued == rescued[position])
        pool = lapvars[same_stratum] if same_stratum.sum() >= _MIN_NEIGHBORS else lapvars[in_window]
        local = config.blur_neighbor_ratio * float(np.median(pool))
        cuts.append(max(local, global_floor))
    return cuts


def _quality_gate(
    stats: list[FrameStats], config: VisionFrontendConfig
) -> tuple[list[int], list[RejectedFrame]]:
    """화질 기준으로 프레임을 걸러낸다.

    Returns:
        (통과한 프레임의 리스트 내 위치, 탈락 기록)
    """
    blur_cuts = _local_blur_cuts(stats, config)
    luma_min, luma_max = config.luma_range

    kept: list[int] = []
    rejected: list[RejectedFrame] = []
    for position, item in enumerate(stats):
        metrics = item.effective_metrics
        if metrics.clipped_low_ratio > config.clipped_ratio_max or metrics.luma_mean < luma_min:
            rejected.append(
                RejectedFrame(
                    frame_index=item.index,
                    timestamp_sec=item.timestamp_sec,
                    reason=RejectReason.UNDEREXPOSED,
                    detail=f"luma={metrics.luma_mean:.3f} clip_low={metrics.clipped_low_ratio:.3f}",
                )
            )
            continue
        if metrics.clipped_high_ratio > config.clipped_ratio_max or metrics.luma_mean > luma_max:
            rejected.append(
                RejectedFrame(
                    frame_index=item.index,
                    timestamp_sec=item.timestamp_sec,
                    reason=RejectReason.OVEREXPOSED,
                    detail=(
                        f"luma={metrics.luma_mean:.3f} clip_high={metrics.clipped_high_ratio:.3f}"
                    ),
                )
            )
            continue
        relative_cut = blur_cuts[position]
        too_blurry = (
            metrics.sharpness_lapvar < config.min_sharpness_lapvar
            or metrics.sharpness_lapvar < relative_cut
        )
        if too_blurry:
            rejected.append(
                RejectedFrame(
                    frame_index=item.index,
                    timestamp_sec=item.timestamp_sec,
                    reason=RejectReason.BLUR,
                    detail=(
                        f"lapvar={metrics.sharpness_lapvar:.2f} 이웃기준={relative_cut:.2f}"
                        f" ({'rescued' if item.rescued else 'normal'})"
                    ),
                )
            )
            continue
        kept.append(position)

    return kept, rejected


def _pick_shot_representatives(
    shots: list[ShotSegment],
    index_to_position: dict[int, int],
    stats: list[FrameStats],
    *,
    sharpness_refs: dict[bool, float],
    text_weight: float,
) -> list[tuple[int, int, float]]:
    """샷마다 점수가 가장 높은 프레임 1장을 고른다.

    Returns:
        (shot_id, 프레임 리스트 내 위치, 점수) 리스트.
    """
    picks: list[tuple[int, int, float]] = []
    for shot in shots:
        best_position = -1
        best_score = -1.0
        for frame_index in shot.frame_indices:
            position = index_to_position[frame_index]
            score = quality.readability_score(
                stats[position].effective_metrics,
                sharpness_ref=sharpness_refs[stats[position].rescued],
                text_weight=text_weight,
            )
            if score > best_score:
                best_score = score
                best_position = position
        picks.append((shot.shot_id, best_position, best_score))
    return picks


def _pick_by_text_detection(
    kept_positions: list[int],
    stats: list[FrameStats],
    sampled: list[tuple[float, Path]],
    cfg: VisionFrontendConfig,
    detector: TextDetector,
) -> tuple[list[tuple[int, int, float]], dict[int, int]]:
    """게이트 통과 프레임에 문자 검출기를 돌려 박스 수 순 + 시간 간격으로 고른다.

    검출은 **게이트가 판정한 것과 같은 이미지**에 한다 — 저조도 구제를 받은 프레임은 구제본
    (WB+감마+CLAHE), 나머지는 원본. 원본에 검출기를 돌리면 어두운 간판이 박스 0개가 되어, 구제로
    게이트를 통과시킨 프레임을 선별에서 다시 버리게 된다. 무거운 보정(디노이즈·언샤프)은 고른
    뒤에만 적용한다.

    Returns:
        ((순번, 프레임 리스트 내 위치, 박스 수) 리스트, {위치: 박스 수}).
    """
    boxes_by_position: dict[int, int] = {}
    for position in kept_positions:
        image = decode.read_bgr(sampled[position][1])
        if stats[position].rescued:
            image, _ = isp.rescue_exposure(image)
        boxes_by_position[position] = detector.detect(image).box_count

    min_gap = budget.temporal_gap_frames(len(stats), cfg.max_keyframes, cfg.text_nms_gap_ratio)
    chosen = budget.select_text_nms(
        [stats[position].index for position in kept_positions],
        [float(boxes_by_position[position]) for position in kept_positions],
        budget=cfg.max_keyframes,
        min_gap=min_gap,
    )
    position_of = {stats[position].index: position for position in kept_positions}
    picks = [
        (order, position_of[index], float(boxes_by_position[position_of[index]]))
        for order, index in enumerate(chosen)
    ]
    logger.info(
        "text_nms_selected",
        detector=detector.name,
        candidates=len(kept_positions),
        min_gap=min_gap,
        selected=len(picks),
        zero_box_selected=sum(1 for _, _, score in picks if score == 0),
    )
    return picks, boxes_by_position


def select_keyframes(
    video_path: Path,
    work_dir: Path,
    *,
    config: VisionFrontendConfig | None = None,
    embedder: Embedder | None = None,
    text_detector: TextDetector | None = None,
) -> KeyframeSelection:
    """영상에서 VLM에 올릴 키프레임을 고른다.

    Args:
        video_path: 원본 영상 파일.
        work_dir: 중간 프레임과 결과 JPEG를 쓸 디렉토리. 호출자가 생성/정리한다.
        config: 선별 설정. None이면 기본값.
        embedder: 프레임 임베더. None이면 `PerceptualEmbedder`(추가 의존성 없음).
        text_detector: `selection_strategy="text_nms"` 에서 쓰는 문자 검출기. None이면
            기본 모델 경로로 만든다(없으면 `TextDetectorUnavailable`).

    Returns:
        선별 결과와 집계. `keyframes[i].path`가 VLM에 넣을 JPEG 경로다.
    """
    active_embedder = embedder or PerceptualEmbedder()
    base_cfg = config or VisionFrontendConfig()
    # 미지정 임계는 임베더 권장값으로 확정한다(유사도 스케일이 임베더마다 다름).
    # 결과의 config에도 실제 사용값이 남도록 확정값으로 교체해 둔다.
    # 임계를 지정하지 않았으면 임베더 권장값에서 출발해, 게이트 통과 프레임의 인접
    # 유사도 분포를 보고 영상별로 한 번 더 낮춘다(아래 4 pass 앞).
    auto_scene_threshold = base_cfg.scene_similarity_threshold is None
    scene_threshold = (
        base_cfg.scene_similarity_threshold
        if base_cfg.scene_similarity_threshold is not None
        else active_embedder.scene_similarity_threshold
    )
    duplicate_threshold = (
        base_cfg.duplicate_shot_threshold
        if base_cfg.duplicate_shot_threshold is not None
        else active_embedder.duplicate_shot_threshold
    )
    cfg = base_cfg.model_copy(
        update={
            "scene_similarity_threshold": scene_threshold,
            "duplicate_shot_threshold": duplicate_threshold,
        }
    )
    started = time.perf_counter()

    frames_dir = work_dir / "frames"
    out_dir = work_dir / "keyframes"
    duration = decode.probe_duration_sec(video_path)
    sampled = decode.extract_frames(
        video_path,
        frames_dir,
        sample_fps=cfg.sample_fps,
        max_frames=cfg.max_decoded_frames,
        long_edge_px=cfg.long_edge_px,
    )

    # --- 1 pass: 저조도 구제 + 지표 + 임베딩 ---
    analyses: list[_FrameAnalysis] = []
    embedding_chunks: list[NDArray[np.float32]] = []
    for start in range(0, len(sampled), _CHUNK_SIZE):
        chunk = sampled[start : start + _CHUNK_SIZE]
        chunk_analyses, embeddings = _analyze_chunk(
            [path for _, path in chunk], active_embedder, cfg
        )
        analyses.extend(chunk_analyses)
        embedding_chunks.append(embeddings)
    all_embeddings = np.vstack(embedding_chunks).astype(np.float32)

    probe = decode.read_bgr(sampled[0][1])
    height, width = probe.shape[:2]
    stats = [
        FrameStats(
            index=index,
            timestamp_sec=timestamp,
            width=width,
            height=height,
            metrics=analyses[index].metrics,
            rescued=analyses[index].rescued,
            metrics_rescued=analyses[index].metrics_rescued,
        )
        for index, (timestamp, _) in enumerate(sampled)
    ]
    rescued_count = sum(1 for item in stats if item.rescued)
    if rescued_count:
        logger.info("frames_rescued", count=rescued_count, of=len(stats))

    lapvars = np.array(
        [item.effective_metrics.sharpness_lapvar for item in stats], dtype=np.float64
    )
    # 샤프닝 판정 기준도 같은 이유로 그룹별로 잡는다. 구제 프레임이 끌어올린 90퍼센타일을
    # 정상 노출 프레임에 적용하면, 이미 선명한 프레임에 언샤프가 걸려 링잉만 늘어난다.
    sharpness_refs = _stratified_percentile(stats, 90.0)

    # --- 2 pass: 화질 게이트 ---
    kept_positions, rejected = _quality_gate(stats, cfg)
    if not kept_positions:
        logger.warning("all_frames_rejected", video=str(video_path))
        kept_positions = [int(np.argmax(lapvars))]
        rejected = [item for item in rejected if item.frame_index != stats[kept_positions[0]].index]

    shots: list[ShotSegment] = []
    if cfg.selection_strategy == "text_nms":
        picks, boxes_by_position = _pick_by_text_detection(
            kept_positions, stats, sampled, cfg, text_detector or DbTextDetector()
        )
        stats = [
            item.model_copy(update={"text_boxes": boxes_by_position[position]})
            if position in boxes_by_position
            else item
            for position, item in enumerate(stats)
        ]
    else:
        # --- 3 pass: 샷 분할 ---
        if auto_scene_threshold:
            scene_threshold = dedup.adaptive_scene_threshold(
                all_embeddings[kept_positions],
                default_threshold=scene_threshold,
                max_cut_rate=cfg.max_shot_cut_rate,
            )
            cfg = cfg.model_copy(update={"scene_similarity_threshold": scene_threshold})

        shots = dedup.segment_shots(
            all_embeddings[kept_positions],
            [stats[position].timestamp_sec for position in kept_positions],
            [stats[position].index for position in kept_positions],
            similarity_threshold=scene_threshold,
            min_shot_gap_sec=cfg.min_shot_gap_sec,
        )
        index_to_position = {stats[position].index: position for position in kept_positions}

        picks = _pick_shot_representatives(
            shots,
            index_to_position,
            stats,
            sharpness_refs=sharpness_refs,
            text_weight=cfg.text_weight,
        )

        # --- 4 pass: 전역 중복 샷 제거 ---
        representative_embeddings = np.vstack(
            [all_embeddings[position] for _, position, _ in picks]
        ).astype(np.float32)
        surviving = dedup.drop_near_duplicate_shots(
            representative_embeddings, similarity_threshold=duplicate_threshold
        )
        dropped_as_duplicate = set(range(len(picks))) - set(surviving)
        for order in sorted(dropped_as_duplicate):
            _, position, _ = picks[order]
            rejected.append(
                RejectedFrame(
                    frame_index=stats[position].index,
                    timestamp_sec=stats[position].timestamp_sec,
                    reason=RejectReason.DUPLICATE,
                    detail=f"앞선 샷과 유사도 >= {duplicate_threshold}",
                )
            )
        picks = [picks[order] for order in surviving]

        # --- 5 pass: 예산 컷 ---
        # 점수 상위 K개가 아니라 장면 다양성 기준으로 남긴다. 점수 순으로 자르면 잘 찍힌
        # 한 장소가 슬롯을 여러 개 먹고 다른 장소가 한 장도 남지 않는다(dedup 모듈 주석 참고).
        if len(picks) > cfg.max_keyframes:
            surviving_embeddings = np.vstack(
                [all_embeddings[position] for _, position, _ in picks]
            ).astype(np.float32)
            keep_orders = set(
                dedup.select_diverse_budget(
                    surviving_embeddings,
                    [score for _, _, score in picks],
                    budget=cfg.max_keyframes,
                )
            )
            for order, (_, position, _) in enumerate(picks):
                if order in keep_orders:
                    continue
                rejected.append(
                    RejectedFrame(
                        frame_index=stats[position].index,
                        timestamp_sec=stats[position].timestamp_sec,
                        reason=RejectReason.BUDGET,
                        detail=f"max_keyframes={cfg.max_keyframes}",
                    )
                )
            picks = [picks[order] for order in sorted(keep_orders)]

    # --- 6 pass: 보정 후 저장 ---
    path_by_index = {index: path for index, (_, path) in enumerate(sampled)}
    keyframes: list[Keyframe] = []
    for shot_id, position, score in picks:
        item = stats[position]
        original = decode.read_bgr(path_by_index[item.index])

        # 보정을 끄더라도 구제된 프레임은 구제본을 내보내야 한다. 원본을 내보내면
        # 게이트 통과 근거(구제 후 지표)와 실제 VLM 입력이 달라진다. 구제본 픽셀은
        # 1 pass에서 버렸으므로 여기서 다시 만든다(WB+감마+CLAHE, 프레임당 수 ms).
        if item.rescued:
            baseline_image, _ = isp.rescue_exposure(original)
            baseline_metrics: QualityMetrics | None = item.metrics_rescued
        else:
            baseline_image, baseline_metrics = original, None

        image = baseline_image
        metrics_after = baseline_metrics
        if cfg.enhance:
            candidate, report = isp.enhance_for_vlm(
                original,
                noise_sigma=item.effective_metrics.noise_sigma,
                sharpness_lapvar=item.effective_metrics.sharpness_lapvar,
                sharpness_ref=sharpness_refs[item.rescued],
            )
            candidate_metrics = quality.compute_metrics(
                candidate, textness=textness.textness(candidate).score
            )
            # 보정이 문자 영역을 지우면 보정을 버린다. 저조도에서 디노이징 강도가
            # 상한에 붙으면 간판 획까지 뭉개지는데, 그 프레임이야말로 잃으면 안 되는
            # 프레임이다(실측: 저조도 장소 1곳이 전체 보정 때문에만 사라졌다).
            reference_textness = item.effective_metrics.textness
            if keep_enhanced(
                candidate_textness=candidate_metrics.textness,
                reference_textness=reference_textness,
                drop_limit=cfg.enhance_textness_drop_limit,
            ):
                image, metrics_after = candidate, candidate_metrics
                logger.debug(
                    "frame_enhanced",
                    frame_index=item.index,
                    stages=report.stages,
                    gamma=round(report.gamma, 3),
                )
            else:
                logger.info(
                    "enhance_rejected",
                    frame_index=item.index,
                    stages=report.stages,
                    textness_before=round(reference_textness, 3),
                    textness_after=round(candidate_metrics.textness, 3),
                )
        out_path = out_dir / f"kf_{item.index:06d}_t{item.timestamp_sec:07.2f}.jpg"
        decode.write_jpeg(image, out_path, quality=cfg.jpeg_quality)
        keyframes.append(
            Keyframe(
                frame_index=item.index,
                shot_id=shot_id,
                timestamp_sec=item.timestamp_sec,
                path=out_path,
                selection_score=score,
                enhanced=cfg.enhance or item.rescued,
                metrics=item.metrics,
                metrics_after=metrics_after,
            )
        )

    decoded = len(sampled)
    selected = len(keyframes)
    summary = SelectionSummary(
        video_duration_sec=duration,
        decoded_frames=decoded,
        rescued_frames=rescued_count,
        after_quality_gate=len(kept_positions),
        shots_detected=len(shots),
        selected_frames=selected,
        reduction_ratio=1.0 - (selected / decoded) if decoded else 0.0,
        embedder=active_embedder.name,
        elapsed_sec=time.perf_counter() - started,
    )
    logger.info(
        "keyframes_selected",
        video=str(video_path),
        decoded=decoded,
        rescued=rescued_count,
        gate=summary.after_quality_gate,
        shots=summary.shots_detected,
        selected=selected,
        reduction=round(summary.reduction_ratio, 4),
        elapsed_sec=round(summary.elapsed_sec, 2),
    )
    return KeyframeSelection(
        keyframes=keyframes,
        shots=shots,
        rejected=rejected,
        frame_stats=stats,
        summary=summary,
        config=cfg,
    )
