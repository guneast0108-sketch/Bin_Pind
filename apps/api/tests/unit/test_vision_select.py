"""문자 saliency와 키프레임 선별 엔드투엔드 테스트.

엔드투엔드 테스트는 ffmpeg으로 짧은 합성 영상을 즉석에서 만들어 돌린다. ffmpeg이 없는
환경에서는 skip 한다.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from app.pipeline.vision import (
    PerceptualEmbedder,
    VisionFrontendConfig,
    compute_metrics,
    select_keyframes,
    textness,
)
from app.pipeline.vision.select import keep_enhanced
from app.pipeline.vision.types import RejectReason
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe 가 필요합니다",
)


# ------------------------------------------------------------------------ textness


def _blank(height: int = 240, width: int = 360, value: int = 140) -> BgrImage:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _with_sign(text: str = "BLUE BOTTLE COFFEE") -> BgrImage:
    image = _blank()
    cv2.rectangle(image, (20, 90), (340, 150), (245, 245, 240), -1)
    cv2.putText(image, text, (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return image


def test_blank_frame_has_no_textness() -> None:
    assert textness(_blank()).score == pytest.approx(0.0)


def test_sign_frame_scores_higher_than_blank() -> None:
    assert textness(_with_sign()).score > textness(_blank()).score


def test_longer_sign_detects_more_regions() -> None:
    short = textness(_with_sign("CAFE"))
    long = textness(_with_sign("CAFE BLUE BOTTLE MANGWON SEOUL"))
    assert long.region_count > short.region_count


def test_textness_boxes_are_within_frame() -> None:
    image = _with_sign()
    result = textness(image)
    height, width = image.shape[:2]
    for box_x, box_y, box_w, box_h in result.boxes:
        assert 0 <= box_x and 0 <= box_y
        assert box_x + box_w <= width
        assert box_y + box_h <= height


def test_textness_score_is_bounded() -> None:
    result = textness(_with_sign("AAAA BBBB CCCC DDDD EEEE FFFF GGGG"))
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.area_ratio <= 1.0


# ----------------------------------------------------------------- 엔드투엔드 선별


def _write_video(path: Path, scenes: list[tuple[BgrImage, int]], fps: int = 10) -> None:
    """(프레임, 초) 목록을 mp4로 인코딩한다."""
    height, width = scenes[0][0].shape[:2]
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame, seconds in scenes:
        payload = frame.tobytes()
        for _ in range(seconds * fps):
            process.stdin.write(payload)
    process.stdin.close()
    assert process.wait() == 0


def _scene(seed: int, height: int = 240, width: int = 360) -> BgrImage:
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = tuple(int(value) for value in rng.integers(60, 200, size=3))
    for _ in range(10):
        x1, y1 = int(rng.integers(0, width - 60)), int(rng.integers(0, height - 60))
        x2, y2 = x1 + int(rng.integers(30, 100)), y1 + int(rng.integers(30, 80))
        color = tuple(int(value) for value in rng.integers(0, 255, size=3))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    return image


def _busy_scene() -> BgrImage:
    """고주파 텍스처가 많은 장면(창문 격자 + 간판)."""
    image = np.full((360, 640, 3), 120, dtype=np.uint8)
    for y in range(20, 340, 40):
        for x in range(20, 620, 60):
            cv2.rectangle(image, (x, y), (x + 40, y + 26), (40, 38, 36), -1)
    cv2.rectangle(image, (150, 150), (500, 220), (245, 245, 240), -1)
    cv2.putText(
        image, "OPEN DAILY", (170, 200), cv2.FONT_HERSHEY_DUPLEX, 1.6, (25, 25, 28), 3, cv2.LINE_AA
    )
    return image


def _calm_scene() -> BgrImage:
    """선명하지만 텍스처가 적은 장면(단색 벽 + 작은 간판)."""
    image = np.full((360, 640, 3), 155, dtype=np.uint8)
    cv2.rectangle(image, (240, 160), (420, 210), (245, 245, 240), -1)
    cv2.putText(
        image, "MUSEUM", (255, 197), cv2.FONT_HERSHEY_DUPLEX, 1.0, (25, 25, 28), 2, cv2.LINE_AA
    )
    return image


@pytest.fixture
def three_scene_video(tmp_path: Path) -> Path:
    """서로 다른 3개 장면이 각 4초씩 이어지는 12초 영상."""
    path = tmp_path / "three_scenes.mp4"
    _write_video(path, [(_scene(1), 4), (_scene(2), 4), (_scene(3), 4)])
    return path


@requires_ffmpeg
def test_select_keyframes_collapses_static_scenes(three_scene_video: Path, tmp_path: Path) -> None:
    selection = select_keyframes(
        three_scene_video,
        tmp_path / "work",
        config=VisionFrontendConfig(
            sample_fps=1.0,
            max_keyframes=8,
            blur_neighbor_ratio=0.0,
            min_shot_gap_sec=1.0,
        ),
        embedder=PerceptualEmbedder(),
    )

    assert selection.summary.decoded_frames >= 10
    # 장면이 3개이므로 최종 프레임은 그 근방이어야 한다(정적 구간이 접혀야 함).
    assert 3 <= selection.summary.selected_frames <= 5
    assert selection.summary.reduction_ratio > 0.5


@requires_ffmpeg
def test_keyframe_files_exist_and_are_readable(three_scene_video: Path, tmp_path: Path) -> None:
    selection = select_keyframes(
        three_scene_video,
        tmp_path / "work",
        config=VisionFrontendConfig(sample_fps=1.0, blur_neighbor_ratio=0.0),
        embedder=PerceptualEmbedder(),
    )
    assert selection.keyframes
    for keyframe in selection.keyframes:
        assert keyframe.path.exists()
        assert cv2.imread(str(keyframe.path)) is not None


@requires_ffmpeg
def test_keyframes_are_time_ordered(three_scene_video: Path, tmp_path: Path) -> None:
    selection = select_keyframes(
        three_scene_video,
        tmp_path / "work",
        config=VisionFrontendConfig(sample_fps=1.0, blur_neighbor_ratio=0.0),
        embedder=PerceptualEmbedder(),
    )
    stamps = [keyframe.timestamp_sec for keyframe in selection.keyframes]
    assert stamps == sorted(stamps)


@requires_ffmpeg
def test_max_keyframes_is_respected(tmp_path: Path) -> None:
    path = tmp_path / "many.mp4"
    _write_video(path, [(_scene(seed), 2) for seed in range(8)])

    selection = select_keyframes(
        path,
        tmp_path / "work",
        config=VisionFrontendConfig(
            sample_fps=1.0, max_keyframes=3, min_shot_gap_sec=1.0, blur_neighbor_ratio=0.0
        ),
        embedder=PerceptualEmbedder(),
    )
    assert selection.summary.selected_frames <= 3
    assert any(item.reason is RejectReason.BUDGET for item in selection.rejected)


@requires_ffmpeg
def test_dark_scene_is_rescued_not_rejected(tmp_path: Path) -> None:
    """저조도 장면은 언더노출로 버려지지 않고 구제되어 살아남아야 한다.

    이 순서가 뒤집히면(게이트 먼저, 보정 나중) 실내에서 찍힌 장소가 전부 사라진다.
    """
    bright = _scene(21)
    dark = (_scene(22).astype(np.float32) * 0.12).astype(np.uint8)
    path = tmp_path / "dark.mp4"
    _write_video(path, [(bright, 4), (dark, 4)])

    selection = select_keyframes(
        path,
        tmp_path / "work",
        config=VisionFrontendConfig(sample_fps=1.0, blur_neighbor_ratio=0.0, min_shot_gap_sec=1.0),
        embedder=PerceptualEmbedder(),
    )

    assert selection.summary.rescued_frames > 0
    assert not any(item.reason is RejectReason.UNDEREXPOSED for item in selection.rejected)

    # 어두운 구간(4초 이후)에서 최소 1장은 선별되어야 한다.
    late = [item for item in selection.keyframes if item.timestamp_sec >= 4.0]
    assert late, "저조도 구간에서 선별된 키프레임이 없습니다"
    assert any(item.rescued for item in selection.frame_stats)


@requires_ffmpeg
def test_summary_counts_are_consistent(three_scene_video: Path, tmp_path: Path) -> None:
    selection = select_keyframes(
        three_scene_video,
        tmp_path / "work",
        config=VisionFrontendConfig(sample_fps=1.0, blur_neighbor_ratio=0.0),
        embedder=PerceptualEmbedder(),
    )
    summary = selection.summary
    assert summary.decoded_frames == len(selection.frame_stats)
    assert summary.selected_frames == len(selection.keyframes)
    assert summary.shots_detected == len(selection.shots)
    assert summary.after_quality_gate <= summary.decoded_frames
    assert summary.selected_frames <= summary.after_quality_gate
    assert 0.0 <= summary.reduction_ratio <= 1.0


class _StrictThresholdEmbedder(PerceptualEmbedder):
    """권장 임계만 다른 임베더 — 임계가 임베더에서 오는지 확인용."""

    @property
    def scene_similarity_threshold(self) -> float:
        return 0.5

    @property
    def duplicate_shot_threshold(self) -> float:
        return 0.99


@requires_ffmpeg
def test_thresholds_default_to_embedder_recommendation(
    three_scene_video: Path, tmp_path: Path
) -> None:
    selection = select_keyframes(
        three_scene_video,
        tmp_path / "work",
        config=VisionFrontendConfig(sample_fps=1.0, blur_neighbor_ratio=0.0),
        embedder=_StrictThresholdEmbedder(),
    )
    assert selection.config.scene_similarity_threshold == 0.5
    assert selection.config.duplicate_shot_threshold == 0.99


@requires_ffmpeg
def test_explicit_thresholds_override_embedder(three_scene_video: Path, tmp_path: Path) -> None:
    selection = select_keyframes(
        three_scene_video,
        tmp_path / "work",
        config=VisionFrontendConfig(
            sample_fps=1.0,
            blur_neighbor_ratio=0.0,
            scene_similarity_threshold=0.8,
            duplicate_shot_threshold=0.9,
        ),
        embedder=_StrictThresholdEmbedder(),
    )
    assert selection.config.scene_similarity_threshold == 0.8
    assert selection.config.duplicate_shot_threshold == 0.9


@requires_ffmpeg
def test_enhance_disabled_keeps_metrics_after_empty(
    three_scene_video: Path, tmp_path: Path
) -> None:
    selection = select_keyframes(
        three_scene_video,
        tmp_path / "work",
        config=VisionFrontendConfig(sample_fps=1.0, enhance=False, blur_neighbor_ratio=0.0),
        embedder=PerceptualEmbedder(),
    )
    # 보정을 끄면 구제되지 않은 프레임은 원본 그대로 나가고 metrics_after 가 비어 있다.
    # 구제된 프레임만 예외적으로 보정본이 나가므로 enhanced=True + metrics_after 존재.
    for keyframe in selection.keyframes:
        if keyframe.enhanced:
            assert keyframe.metrics_after is not None
        else:
            assert keyframe.metrics_after is None


# ------------------------------------------ 화질 게이트: 이웃 기준 (회귀)


@requires_ffmpeg
def test_low_texture_scene_is_not_rejected_as_blur(tmp_path: Path) -> None:
    """텍스처가 적은 장면이 '흐리다'고 탈락해서는 안 된다.

    전역 선명도 분포의 하위 N%를 자르면 "흔들린 프레임이 항상 N% 있다"고 가정하는 셈이다.
    흔들림이 전혀 없고 장면마다 텍스처 양만 다른 영상에서, 가장 단조로운 장면이 구간
    전체로 탈락했다(실측: 장소 1곳 손실). 흔들림은 시간적 아티팩트이므로 기준은
    이웃 프레임이어야 한다.
    """
    path = tmp_path / "mixed_texture.mp4"
    _write_video(path, [(_busy_scene(), 4), (_calm_scene(), 4), (_busy_scene(), 4)])

    selection = select_keyframes(
        path,
        tmp_path / "work",
        config=VisionFrontendConfig(sample_fps=1.0, min_shot_gap_sec=1.0, enhance=False),
        embedder=PerceptualEmbedder(),
    )

    calm_window = [item for item in selection.frame_stats if 4.0 <= item.timestamp_sec <= 7.0]
    assert calm_window, "단조로운 장면 구간이 샘플링되지 않았다"
    blurred = {
        item.timestamp_sec for item in selection.rejected if item.reason is RejectReason.BLUR
    }
    assert not [item for item in calm_window if item.timestamp_sec in blurred]
    # 그리고 그 장면이 실제로 키프레임으로 남아야 한다.
    assert any(4.0 <= keyframe.timestamp_sec <= 7.0 for keyframe in selection.keyframes)


@requires_ffmpeg
def test_motion_blurred_frame_is_still_rejected(tmp_path: Path) -> None:
    """이웃 기준으로 바꿔도 진짜 흔들린 프레임은 걸러야 한다."""
    sharp = _busy_scene()
    smeared = np.asarray(cv2.GaussianBlur(sharp, (0, 0), 6.0), dtype=np.uint8)
    path = tmp_path / "with_blur.mp4"
    _write_video(path, [(sharp, 3), (smeared, 1), (sharp, 3)])

    selection = select_keyframes(
        path,
        tmp_path / "work",
        config=VisionFrontendConfig(sample_fps=2.0, min_shot_gap_sec=0.5, enhance=False),
        embedder=PerceptualEmbedder(),
    )

    assert any(item.reason is RejectReason.BLUR for item in selection.rejected)


# --------------------------------- 보정 회귀 가드: 문자 영역이 줄면 보정을 버린다


def _dark_noisy(image: BgrImage, *, gain: float = 0.02, seed: int = 0) -> BgrImage:
    """선형 광량 도메인에서 노출을 줄이고 샷/리드 노이즈를 얹는다.

    sRGB 값을 그냥 곱하면 노이즈 없이 깔끔하게 어두운 프레임이 되고, 그건 실제 저조도가
    아니다. 광량이 줄면 샷 노이즈의 상대 크기가 커지는 것이 요점이다.
    """
    rng = np.random.default_rng(seed)
    full_well = 3000.0
    linear = np.power(image.astype(np.float32) / 255.0, 2.2) * gain
    electrons = np.asarray(rng.poisson(linear * full_well), dtype=np.float32)
    electrons += rng.normal(0.0, 6.0, electrons.shape).astype(np.float32)
    measured = np.clip(electrons / full_well, 0.0, 1.0).astype(np.float32)
    darkened = np.asarray(np.power(measured, 1 / 2.2) * 255.0, dtype=np.uint8)
    return darkened


def test_keep_enhanced_rejects_textness_regression() -> None:
    """보정이 문자 영역을 줄이면 되돌려야 한다."""
    assert not keep_enhanced(candidate_textness=0.88, reference_textness=1.0, drop_limit=0.03)
    assert keep_enhanced(candidate_textness=0.99, reference_textness=1.0, drop_limit=0.03)
    assert keep_enhanced(candidate_textness=1.0, reference_textness=0.52, drop_limit=0.03)


def test_keep_enhanced_drop_limit_zero_is_strict() -> None:
    assert keep_enhanced(candidate_textness=1.0, reference_textness=1.0, drop_limit=0.0)
    assert not keep_enhanced(candidate_textness=0.999, reference_textness=1.0, drop_limit=0.0)


@requires_ffmpeg
def test_saved_keyframe_matches_reported_metrics(tmp_path: Path) -> None:
    """디스크에 쓴 프레임과 보고된 보정 후 지표가 같아야 한다.

    구제·보정·되돌림 경로가 세 갈래여서, 한 갈래에서 "다른 이미지를 저장하고 다른
    지표를 보고하는" 불일치가 나기 쉽다. 그러면 게이트 통과 근거와 실제 VLM 입력이
    달라지고, 벤치마크 숫자가 조용히 거짓이 된다.
    """
    path = tmp_path / "dark_sign.mp4"
    _write_video(path, [(_dark_noisy(_busy_scene()), 4)])

    selection = select_keyframes(
        path,
        tmp_path / "work",
        config=VisionFrontendConfig(sample_fps=1.0, min_shot_gap_sec=1.0),
        embedder=PerceptualEmbedder(),
    )

    assert selection.keyframes
    for keyframe in selection.keyframes:
        assert keyframe.metrics_after is not None
        raw = cv2.imread(str(keyframe.path))
        assert raw is not None
        written = np.asarray(raw, dtype=np.uint8)
        measured = compute_metrics(written, textness=textness(written).score)
        # JPEG 재압축 때문에 정확히 같지는 않으므로 느슨한 허용오차를 둔다.
        assert measured.luma_mean == pytest.approx(keyframe.metrics_after.luma_mean, abs=0.02)
        assert measured.textness == pytest.approx(keyframe.metrics_after.textness, abs=0.1)


@requires_ffmpeg
def test_sustained_blur_segment_is_rejected_by_global_floor(tmp_path: Path) -> None:
    """수 초간 이어지는 흔들림 구간은 이웃 기준으로 잡히지 않는다.

    이웃도 똑같이 흐리면 상대 기준은 통과한다. 그래서 영상 전체 중앙값에서 잡은 약한
    절대 하한을 함께 둔다. 실측에서 lapvar 9.8(사실상 형체 없음)인 프레임이 이 경로로
    키프레임에 남았다.
    """
    sharp = _busy_scene()
    smeared = np.asarray(cv2.GaussianBlur(sharp, (0, 0), 8.0), dtype=np.uint8)
    path = tmp_path / "long_blur.mp4"
    _write_video(path, [(sharp, 4), (smeared, 4), (sharp, 4)])

    selection = select_keyframes(
        path,
        tmp_path / "work",
        config=VisionFrontendConfig(sample_fps=1.0, min_shot_gap_sec=1.0, enhance=False),
        embedder=PerceptualEmbedder(),
    )

    blurred_window = {
        item.timestamp_sec
        for item in selection.rejected
        if item.reason is RejectReason.BLUR and 4.0 <= item.timestamp_sec <= 7.0
    }
    assert blurred_window, "이어지는 흔들림 구간이 통째로 통과했다"
    assert not any(4.0 <= keyframe.timestamp_sec <= 7.0 for keyframe in selection.keyframes)
