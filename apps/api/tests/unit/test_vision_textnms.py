"""학습된 문자 검출기 + 시간 간격 예산 컷 (selection_strategy="text_nms")."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from app.pipeline.vision import (
    PerceptualEmbedder,
    TextDetection,
    VisionFrontendConfig,
    select_keyframes,
)
from app.pipeline.vision.budget import select_text_nms, temporal_gap_frames
from app.pipeline.vision.textdet import DbTextDetector, default_model_path, postprocess
from numpy.typing import NDArray

from tests.unit.test_vision_select import _scene, _write_video

BgrImage = NDArray[np.uint8]

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe 가 필요합니다",
)
requires_model = pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None or not default_model_path().exists(),
    reason="onnxruntime 과 검출 모델이 필요합니다 (python -m app.pipeline.vision.textdet --fetch)",
)


# ------------------------------------------------------------------------ 예산 컷


def test_gap_is_half_of_even_spacing() -> None:
    # 392장에 16장이면 고르게 폈을 때 간격 24.5 → 절반 12
    assert temporal_gap_frames(392, 16, 0.5) == 12
    assert temporal_gap_frames(0, 16, 0.5) == 0


def test_highest_scores_win_but_neighbours_are_suppressed() -> None:
    indices = list(range(10))
    scores = [0, 9, 8, 0, 0, 0, 7, 0, 0, 5]
    # 1(9) 선택 → 2(8)는 간격 1 ≤ 2 라 제외 → 6(7) → 9(5)
    assert select_text_nms(indices, scores, budget=5, min_gap=2) == [1, 6, 9]


def test_budget_caps_selection_and_output_is_time_ordered() -> None:
    indices = [30, 0, 20, 10]
    scores = [1.0, 4.0, 3.0, 2.0]
    assert select_text_nms(indices, scores, budget=2, min_gap=0) == [0, 20]


def test_ties_prefer_earlier_frame() -> None:
    assert select_text_nms([5, 2, 9], [1.0, 1.0, 1.0], budget=1, min_gap=0) == [2]


def test_gap_can_leave_budget_unfilled() -> None:
    assert select_text_nms(list(range(5)), [1.0] * 5, budget=5, min_gap=10) == [0]


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        select_text_nms([1, 2], [1.0], budget=1, min_gap=0)


# ------------------------------------------------------------------------ 후처리


def test_postprocess_counts_confident_boxes_only() -> None:
    prob = np.zeros((200, 300), dtype=np.float32)
    prob[20:40, 20:120] = 0.9  # 문자열 1
    prob[80:100, 50:200] = 0.8  # 문자열 2
    prob[150:170, 30:90] = 0.35  # 이진화는 넘지만 평균 확률 0.5 미만 → 제외
    prob[180:182, 200:260] = 0.9  # 짧은 변 2px → 제외
    result = postprocess(prob, (200, 300))
    assert result.box_count == 2
    assert 0.0 < result.area_ratio < 1.0


def test_postprocess_empty_map() -> None:
    result = postprocess(np.zeros((64, 64), dtype=np.float32), (64, 64))
    assert result == TextDetection(box_count=0, area_ratio=0.0)


@requires_model
def test_real_detector_finds_rendered_text() -> None:
    detector = DbTextDetector()
    blank = np.full((360, 640, 3), 128, dtype=np.uint8)
    sign = blank.copy()
    cv2.rectangle(sign, (60, 120), (580, 230), (245, 245, 240), -1)
    cv2.putText(sign, "BLUE BOTTLE", (80, 200), cv2.FONT_HERSHEY_DUPLEX, 2.2, (20, 20, 20), 4)
    assert detector.detect(blank).box_count == 0
    assert detector.detect(sign).box_count >= 1


# ------------------------------------------------------------------------ 엔드투엔드


class _MarkerDetector:
    """흰 간판 판(245)이 있으면 박스 5개, 없으면 0개를 돌려주는 가짜 검출기."""

    name = "fake-marker"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: BgrImage) -> TextDetection:
        self.calls += 1
        white = float(np.mean(np.all(image > 235, axis=2)))
        return TextDetection(box_count=5 if white > 0.05 else 0, area_ratio=white)


def _sign_scene(seed: int) -> BgrImage:
    image = _scene(seed)
    cv2.rectangle(image, (40, 80), (320, 160), (245, 245, 245), -1)
    return image


@pytest.fixture
def mixed_video(tmp_path: Path) -> Path:
    """간판 없는 장면 사이에 간판 장면 3개가 끼어 있는 24초 영상 (장면당 3초)."""
    scenes = [
        (_scene(10), 3),
        (_sign_scene(11), 3),
        (_scene(12), 3),
        (_scene(13), 3),
        (_sign_scene(14), 3),
        (_scene(15), 3),
        (_scene(16), 3),
        (_sign_scene(17), 3),
    ]
    path = tmp_path / "mixed.mp4"
    _write_video(path, scenes)
    return path


@requires_ffmpeg
def test_text_nms_picks_sign_scenes(mixed_video: Path, tmp_path: Path) -> None:
    detector = _MarkerDetector()
    selection = select_keyframes(
        mixed_video,
        tmp_path / "work",
        config=VisionFrontendConfig(max_keyframes=3, selection_strategy="text_nms"),
        embedder=PerceptualEmbedder(),
        text_detector=detector,
    )
    sign_windows = [(3, 6), (12, 15), (21, 24)]
    chosen = [keyframe.timestamp_sec for keyframe in selection.keyframes]
    assert len(chosen) == 3
    for (start, end), timestamp in zip(sign_windows, chosen, strict=True):
        assert start <= timestamp < end
    # 게이트를 통과한 프레임에만 검출기를 돌리고, 그 결과를 frame_stats 에 남긴다
    filled = [item.text_boxes for item in selection.frame_stats if item.text_boxes is not None]
    assert detector.calls == len(filled) > 0
    assert selection.shots == []


@requires_ffmpeg
def test_default_strategy_does_not_call_detector(mixed_video: Path, tmp_path: Path) -> None:
    detector = _MarkerDetector()
    selection = select_keyframes(
        mixed_video,
        tmp_path / "work",
        config=VisionFrontendConfig(max_keyframes=3),
        embedder=PerceptualEmbedder(),
        text_detector=detector,
    )
    assert detector.calls == 0
    assert all(item.text_boxes is None for item in selection.frame_stats)
    assert selection.shots
