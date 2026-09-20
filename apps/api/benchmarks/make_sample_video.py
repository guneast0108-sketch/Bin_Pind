"""벤치마크용 합성 브이로그 영상 생성.

실제 YouTube 영상은 저작권·재현성 문제가 있어 레포에 넣을 수 없다. 대신 비전 프론트엔드가
다뤄야 하는 조건을 의도적으로 심은 합성 영상을 만들어, 누가 클론해도 같은 수치를 얻게 한다.

    python benchmarks/make_sample_video.py sample_vlog.mp4
    python benchmarks/bench_keyframes.py sample_vlog.mp4

재현하는 브이로그 특성:
  - 한 장소에 여러 초 머무름(거의 동일한 프레임이 연속)
  - 간판/메뉴판 텍스트가 있는 컷
  - 핸드헬드 모션 블러 컷
  - 실내 저조도 컷
  - 같은 장소 재방문 컷
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]

W, H = 1280, 720
FPS = 30
RNG = np.random.default_rng(20260914)

SCENES = [
    # (label, seconds, signage_text, base_bgr, dark, blur_sigma)
    ("street_cafe", 8, "BLUE BOTTLE COFFEE  SEOUL", (150, 160, 170), False, 0.0),
    ("cafe_menu", 6, "AMERICANO 4500  LATTE 5000  CAKE 6500", (200, 205, 210), False, 0.0),
    ("walk_blur", 4, "", (140, 150, 160), False, 6.0),
    ("gallery_dark", 7, "SEOUL MUSEUM OF ART", (40, 42, 45), True, 0.0),
    ("park", 6, "", (90, 150, 110), False, 0.0),
    ("shop_sign", 5, "OLD FERRY DONUT  MANGWON", (170, 150, 130), False, 0.0),
    ("walk_blur2", 3, "", (150, 150, 150), False, 8.0),
    ("street_cafe", 5, "BLUE BOTTLE COFFEE  SEOUL", (150, 160, 170), False, 0.0),
    ("restaurant", 6, "GEUMDWAEJI SIKDANG", (120, 100, 95), False, 0.0),
    ("overexposed_sky", 3, "", (250, 252, 253), False, 0.0),
]


_BG_CACHE: dict[str, BgrImage] = {}


def scene_background(label: str, text: str, base: tuple[int, int, int]) -> BgrImage:
    """장면별 고정 배경. 같은 label은 같은 배경을 재사용한다(재방문 컷 재현)."""
    if label in _BG_CACHE:
        return _BG_CACHE[label]
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:, :] = base

    # 장면마다 다른 구조물(건물/벽/창) — 임베딩이 장면을 구분할 근거
    scene_rng = np.random.default_rng(abs(hash(label)) % 100000)
    for _ in range(14):
        x1 = int(scene_rng.integers(0, W - 160))
        y1 = int(scene_rng.integers(0, H - 160))
        x2 = x1 + int(scene_rng.integers(80, 300))
        y2 = y1 + int(scene_rng.integers(80, 260))
        color = tuple(int(value) for value in scene_rng.integers(20, 235, size=3))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

    if text:
        # 간판: 밝은 판 위에 어두운 글자
        cv2.rectangle(frame, (90, 470), (1190, 610), (245, 245, 240), -1)
        cv2.rectangle(frame, (90, 470), (1190, 610), (30, 30, 30), 3)
        cv2.putText(
            frame, text, (110, 555), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (25, 25, 25), 3, cv2.LINE_AA
        )

    _BG_CACHE[label] = frame
    background: BgrImage = frame
    return background


def make_frame(label: str, text: str, base: tuple[int, int, int], jitter: int) -> BgrImage:
    """캐시된 배경에 프레임마다 미세한 카메라 흔들림과 센서 노이즈를 더한다."""
    frame = scene_background(label, text, base)

    # 카메라 흔들림 (평행이동)
    if jitter:
        matrix = np.array(
            [
                [1.0, 0.0, float(RNG.integers(-jitter, jitter + 1))],
                [0.0, 1.0, float(RNG.integers(-jitter, jitter + 1))],
            ],
            dtype=np.float32,
        )
        frame = np.asarray(
            cv2.warpAffine(frame, matrix, (W, H), borderMode=cv2.BORDER_REFLECT), np.uint8
        )

    # 센서 노이즈
    noise = RNG.normal(0.0, 3.0, frame.shape).astype(np.float32)
    noisy = np.asarray(np.clip(frame.astype(np.float32) + noise, 0, 255), dtype=np.uint8)
    return noisy


def main(out_path: Path) -> None:
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
            f"{W}x{H}",
            "-framerate",
            str(FPS),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None

    counter = 0
    for label, seconds, text, base, dark, blur_sigma in SCENES:
        for _ in range(seconds * FPS):
            frame = make_frame(label, text, base, jitter=3 if blur_sigma else 1)
            if blur_sigma > 0:
                frame = np.asarray(cv2.GaussianBlur(frame, (0, 0), blur_sigma), np.uint8)
            if dark:
                frame = np.asarray(np.clip(frame.astype(np.float32) * 0.22, 0, 255), np.uint8)
                frame = np.asarray(
                    np.clip(frame.astype(np.float32) + RNG.normal(0.0, 9.0, frame.shape), 0, 255),
                    np.uint8,
                )
                # 실내 백열등: 파란 채널을 죽여 색온도를 따뜻하게 (WB 보정 대상)
                frame[:, :, 0] = (frame[:, :, 0].astype(np.float32) * 0.55).astype(np.uint8)
            process.stdin.write(frame.tobytes())
            counter += 1

    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"wrote {out_path} ({counter} frames, {counter / FPS:.1f}s)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
