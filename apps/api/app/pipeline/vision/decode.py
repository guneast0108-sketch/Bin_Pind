"""프레임 디코딩.

`ffmpeg`으로 균일 간격 샘플링 후 BGR uint8 배열로 읽어들인다.
파이프라인 규칙에 따라 순수 함수이며, 임시 파일은 호출자가 준 디렉토리 안에서만 만든다.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import structlog
from numpy.typing import NDArray

logger = structlog.get_logger(__name__)

BgrImage = NDArray[np.uint8]


class FfmpegNotAvailable(RuntimeError):
    """ffmpeg/ffprobe 실행 파일을 찾을 수 없음."""


class VideoDecodeFailed(RuntimeError):
    """디코딩 결과가 비어 있거나 ffmpeg이 실패함."""


def probe_duration_sec(video_path: Path) -> float:
    """ffprobe로 영상 길이를 초 단위로 읽는다. 실패 시 0.0."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True, timeout=60).stdout
    except FileNotFoundError as exc:  # pragma: no cover - 환경 의존
        raise FfmpegNotAvailable("ffprobe 를 찾을 수 없습니다") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.warning("ffprobe_failed", video=str(video_path))
        return 0.0

    try:
        return float(json.loads(raw)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        logger.warning("ffprobe_duration_unparsable", video=str(video_path))
        return 0.0


def _scale_filter(long_edge_px: int) -> str:
    """긴 변을 long_edge_px 로 맞추는 ffmpeg scale 필터. 확대는 하지 않는다."""
    return (
        f"scale=if(gte(iw\\,ih)\\,min({long_edge_px}\\,iw)\\,-2)"
        f":if(lt(iw\\,ih)\\,min({long_edge_px}\\,ih)\\,-2)"
    )


def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    sample_fps: float = 1.0,
    max_frames: int = 900,
    long_edge_px: int = 1280,
) -> list[tuple[float, Path]]:
    """균일 간격으로 프레임을 뽑아 PNG로 저장한다.

    Args:
        video_path: 원본 영상.
        out_dir: 프레임을 저장할 디렉토리(호출자가 생성/정리 책임).
        sample_fps: 초당 샘플 수.
        max_frames: 저장할 최대 프레임 수.
        long_edge_px: 긴 변 리사이즈 목표.

    Returns:
        (timestamp_sec, png_path) 리스트. 시간 순서.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.png"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={sample_fps},{_scale_filter(long_edge_px)}",
        "-frames:v",
        str(max_frames),
        "-vsync",
        "0",
        str(pattern),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=1800)
    except FileNotFoundError as exc:  # pragma: no cover - 환경 의존
        raise FfmpegNotAvailable("ffmpeg 를 찾을 수 없습니다") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace")[:500]
        raise VideoDecodeFailed(f"ffmpeg 실패: {stderr}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoDecodeFailed("ffmpeg 타임아웃(30분)") from exc

    paths = sorted(out_dir.glob("frame_*.png"))
    if not paths:
        raise VideoDecodeFailed("디코딩된 프레임이 없습니다")

    step = 1.0 / sample_fps
    result = [(idx * step, path) for idx, path in enumerate(paths)]
    logger.info(
        "frames_extracted",
        video=str(video_path),
        count=len(result),
        sample_fps=sample_fps,
        long_edge_px=long_edge_px,
    )
    return result


def read_bgr(path: Path) -> BgrImage:
    """이미지를 BGR uint8로 읽는다."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise VideoDecodeFailed(f"이미지를 읽을 수 없습니다: {path}")
    return image.astype(np.uint8)


def write_jpeg(image: BgrImage, path: Path, *, quality: int = 92) -> None:
    """BGR 이미지를 JPEG로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise VideoDecodeFailed(f"JPEG 저장 실패: {path}")
