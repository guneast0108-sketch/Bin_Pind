"""장소 보존률(recall) 검증용 장면 생성.

## 왜 별도 생성기가 필요한가

`make_sample_video.py`는 파이프라인이 돌아가는지 보는 스모크 영상이다. 하지만
"프레임을 얼마나 줄였는가"는 이 파이프라인의 목적이 아니다. 목적은 **줄이면서도
장소를 잃지 않는 것**이고, 그걸 재려면 프레임마다 "여기에 어떤 장소가 있었는지"가
정답으로 있어야 한다. 실제 여행 영상에는 그 정답이 붙어 있지 않다.

그래서 정답을 아는 장면을 만든다. 장소마다 고유한 간판 문자열을 박고, 어느 구간이
어느 장소인지를 JSON으로 함께 내보낸다. 평가는 `validate_recall.py`가 맡고,
판정은 파이프라인 자신의 textness 점수가 아니라 **외부 OCR(tesseract)**이 한다.

## 어려움을 어디에 넣었나

- **저조도 장소** — 실내·야간 간판. sRGB에서 밝기만 낮추는 게 아니라 선형 광량
  도메인에서 노출을 줄이고 샷 노이즈(Poisson) + 리드 노이즈(Gaussian)를 얹는다.
  실제 저조도 프레임이 어두운 동시에 노이즈가 많은 이유가 이것이다. 보정 없이는
  OCR이 못 읽고, 미니 ISP가 동작해야 읽힌다.
- **재방문** — 같은 장소를 나중에 다시 지나간다. 중복 제거가 이걸 접어야 하지만,
  장소를 잃어서는 안 된다. 접는 것과 잃는 것은 다르다.
- **흔들림** — 일부 프레임에 모션 블러. 화질 게이트가 걸러야 하는 쪽이다.
- **핸드헬드 팬** — 크롭 윈도우가 큰 캔버스를 천천히 지나가며 미세 지터가 붙는다.

## 사용

    python benchmarks/make_recall_scene.py --out-dir scene_out

    scene_out/scene.mp4          입력 영상
    scene_out/ground_truth.json  구간별 정답 (장소 id, 간판 문자열, 저조도 여부)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

# sRGB 전달함수 근사. 정확한 구간 선형 sRGB가 아니라 감마 2.2 근사를 쓴다 —
# 저조도 열화의 성격(어두움 + 노이즈)을 재현하는 데는 충분하다.
GAMMA = 2.2

# 센서 모델 상수. 값 자체보다 "밝기를 낮추면 노이즈가 같이 늘어난다"는 관계가 중요하다.
FULL_WELL_ELECTRONS = 3000.0
READ_NOISE_ELECTRONS = 6.0

# 저조도 장소의 노출 배율. 보정 없이는 OCR이 못 읽고, 미니 ISP가 동작하면 읽히는
# 구간을 노린 값이다. 이보다 더 낮추면 신호 자체가 남지 않아 "복원 불가" 판정이 맞다.
DARK_GAIN = 0.02


@dataclass(frozen=True)
class Place:
    """정답을 아는 장소 하나."""

    place_id: str
    sign_text: str
    hue: int
    dark: bool = False
    revisit: bool = False


@dataclass
class Segment:
    """타임라인의 한 구간 = 장소 하나를 지나가는 동안."""

    place: Place
    start_sec: float
    end_sec: float
    pan_from: float
    exposure: float = 1.0
    blur_frames: tuple[int, ...] = field(default=())


PLACES: tuple[Place, ...] = (
    Place("p1", "CAFE LUMIERE", hue=12),
    Place("p2", "HOTEL ASAHI", hue=105, dark=True),
    Place("p3", "RAMEN DAIKOKU", hue=8),
    Place("p4", "BOOKS TANAKA", hue=30, dark=True),
    Place("p5", "BAR MIDNIGHT", hue=140, dark=True),
    Place("p6", "SAKURA BAKERY", hue=170),
    Place("p7", "MUSEUM NORTH", hue=95),
    Place("p8", "GREEN PHARMACY", hue=60),
)

# 나중에 다시 지나가는 장소. 중복 제거가 접어야 하는 쪽.
REVISITS: tuple[str, ...] = ("p1", "p3", "p7")


def _srgb_to_linear(image: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.power(np.clip(image, 0.0, 1.0), GAMMA, dtype=np.float32)


def _linear_to_srgb(image: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.power(np.clip(image, 0.0, 1.0), 1.0 / GAMMA, dtype=np.float32)


def build_facade(
    place: Place, size: tuple[int, int], rng: np.random.Generator
) -> NDArray[np.uint8]:
    """장소 하나의 큰 캔버스를 그린다. 크롭 윈도우가 이 위를 지나간다."""
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # 하늘 — 위에서 아래로 밝아지는 그라디언트
    sky_height = int(height * 0.28)
    gradient = np.linspace(150, 205, sky_height, dtype=np.float32)
    canvas[:sky_height] = gradient[:, None, None]

    # 건물 벽 — 장소마다 다른 색상(HSV로 만들고 BGR로 변환)
    wall = np.zeros((height - sky_height, width, 3), dtype=np.uint8)
    wall[:, :, 0] = place.hue
    wall[:, :, 1] = 70
    wall[:, :, 2] = 150
    canvas[sky_height:] = cv2.cvtColor(wall, cv2.COLOR_HSV2BGR)

    # 벽돌 줄눈 — 고주파 성분이 전혀 없으면 선명도 지표가 0에 붙는다
    for y in range(sky_height, height, 26):
        cv2.line(canvas, (0, y), (width, y), (90, 90, 95), 1)
    for x in range(0, width, 52):
        cv2.line(canvas, (x, sky_height), (x, height), (90, 90, 95), 1)

    # 창문
    for row in range(3):
        for column in range(width // 220):
            x = 60 + column * 220
            y = sky_height + 40 + row * 150
            cv2.rectangle(canvas, (x, y), (x + 120, y + 100), (58, 52, 48), -1)
            cv2.rectangle(canvas, (x, y), (x + 120, y + 100), (188, 188, 190), 3)
            cv2.line(canvas, (x + 60, y), (x + 60, y + 100), (188, 188, 190), 2)

    # 간판 — 이 영상에서 장소를 식별하는 유일한 단서
    board_width = int(width * 0.45)
    board_height = 190
    bx = (width - board_width) // 2
    by = sky_height + 24
    cv2.rectangle(canvas, (bx, by), (bx + board_width, by + board_height), (242, 240, 236), -1)
    cv2.rectangle(canvas, (bx, by), (bx + board_width, by + board_height), (40, 40, 44), 5)

    scale = 3.2
    thickness = 6
    (text_width, text_height), _ = cv2.getTextSize(
        place.sign_text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness
    )
    while text_width > board_width - 56 and scale > 0.8:
        scale -= 0.1
        (text_width, text_height), _ = cv2.getTextSize(
            place.sign_text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness
        )
    cv2.putText(
        canvas,
        place.sign_text,
        (bx + (board_width - text_width) // 2, by + (board_height + text_height) // 2),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        (26, 26, 30),
        thickness,
        cv2.LINE_AA,
    )

    # 보도와 차도
    ground = int(height * 0.86)
    cv2.rectangle(canvas, (0, ground), (width, height), (128, 128, 132), -1)
    cv2.line(canvas, (0, ground), (width, ground), (70, 70, 74), 3)

    # 센서 고정 패턴처럼 아주 약한 텍스처. 완전 평탄한 면을 없앤다.
    grain = rng.normal(0.0, 3.0, canvas.shape).astype(np.float32)
    textured: NDArray[np.uint8] = np.clip(canvas.astype(np.float32) + grain, 0, 255).astype(
        np.uint8
    )
    return textured


def apply_low_light(
    frame: NDArray[np.uint8], gain: float, rng: np.random.Generator
) -> NDArray[np.uint8]:
    """선형 광량 도메인에서 노출을 줄이고 센서 노이즈를 얹는다.

    sRGB 값을 그냥 곱해서 어둡게 만들면 노이즈 없이 깔끔하게 어두운 프레임이 나오고,
    그건 실제 저조도가 아니다. 광량이 줄면 샷 노이즈의 상대 크기가 커지는 것이 핵심이다.
    """
    linear = _srgb_to_linear(frame.astype(np.float32) / 255.0) * gain
    electrons = linear * FULL_WELL_ELECTRONS
    shot = np.asarray(rng.poisson(np.clip(electrons, 0.0, None)), dtype=np.float32)
    read = rng.normal(0.0, READ_NOISE_ELECTRONS, shot.shape).astype(np.float32)
    measured = ((shot + read) / FULL_WELL_ELECTRONS).astype(np.float32)
    degraded: NDArray[np.uint8] = (_linear_to_srgb(measured) * 255.0).astype(np.uint8)
    return degraded


def _motion_blur(frame: NDArray[np.uint8], length: int = 17) -> NDArray[np.uint8]:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    blurred = np.asarray(cv2.filter2D(frame, -1, kernel), dtype=np.uint8)
    return blurred


def build_timeline(seconds_per_segment: float) -> list[Segment]:
    """장소를 한 번씩 지나가고, 일부를 나중에 다시 지나가는 타임라인."""
    by_id = {place.place_id: place for place in PLACES}
    segments: list[Segment] = []
    cursor = 0.0

    for index, place in enumerate(PLACES):
        segments.append(
            Segment(
                place=place,
                start_sec=cursor,
                end_sec=cursor + seconds_per_segment,
                pan_from=0.0 if index % 2 == 0 else 0.25,
                # 두 장소에서는 손이 흔들린다 — 화질 게이트가 걸러야 하는 쪽
                blur_frames=(2, 3) if place.place_id in {"p3", "p6"} else (),
            )
        )
        cursor += seconds_per_segment

    for place_id in REVISITS:
        place = by_id[place_id]
        segments.append(
            Segment(
                place=Place(
                    place_id=place.place_id,
                    sign_text=place.sign_text,
                    hue=place.hue,
                    dark=place.dark,
                    revisit=True,
                ),
                start_sec=cursor,
                end_sec=cursor + seconds_per_segment,
                pan_from=0.4,
                exposure=1.12,  # 같은 장소지만 노출이 조금 다르다
            )
        )
        cursor += seconds_per_segment

    return segments


def render(out_dir: Path, *, fps: int, seconds_per_segment: float, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_width, frame_height = 1280, 720
    # 캔버스 폭은 간판이 팬 구간 내내 크롭 안에 완전히 들어오도록 정했다. 간판이
    # 잘려 나가면 OCR이 못 읽는 것이 파이프라인 탓인지 장면 탓인지 구분할 수 없다.
    canvas_size = (1700, 900)
    segments = build_timeline(seconds_per_segment)

    facades = {
        segment.place.place_id: build_facade(segment.place, canvas_size, rng)
        for segment in segments
    }

    video_path = out_dir / "scene.mp4"
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():  # pragma: no cover - 환경 의존
        raise SystemExit("VideoWriter를 열 수 없습니다 (mp4v 코덱 확인)")

    frames_per_segment = int(round(seconds_per_segment * fps))
    max_pan = canvas_size[0] - frame_width

    for segment in segments:
        facade = facades[segment.place.place_id]
        for local in range(frames_per_segment):
            progress = local / max(frames_per_segment - 1, 1)
            # 팬: 시작 위치에서 캔버스를 천천히 지나간다
            pan = (segment.pan_from + 0.35 * progress) * max_pan
            jitter_x, jitter_y = rng.normal(0.0, 2.5, 2)
            x = int(np.clip(pan + jitter_x, 0, max_pan))
            y_center = (canvas_size[1] - frame_height) / 2
            y = int(np.clip(y_center + jitter_y, 0, canvas_size[1] - frame_height))
            frame = facade[y : y + frame_height, x : x + frame_width].copy()

            if local in segment.blur_frames:
                frame = _motion_blur(frame)

            if segment.place.dark:
                frame = apply_low_light(frame, DARK_GAIN * segment.exposure, rng)
            elif segment.exposure != 1.0:
                frame = np.clip(frame.astype(np.float32) * segment.exposure, 0, 255).astype(
                    np.uint8
                )

            writer.write(frame)

    writer.release()

    truth: dict[str, object] = {
        "video": video_path.name,
        "fps": fps,
        "places": [
            {
                "place_id": place.place_id,
                "sign_text": place.sign_text,
                "dark": place.dark,
                "segments": [
                    [segment.start_sec, segment.end_sec]
                    for segment in segments
                    if segment.place.place_id == place.place_id
                ],
            }
            for place in PLACES
        ],
    }
    truth_path = out_dir / "ground_truth.json"
    truth_path.write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")

    dark = sum(1 for place in PLACES if place.dark)
    total_sec = len(segments) * seconds_per_segment
    print(f"영상: {video_path}  ({len(segments)}구간 · {total_sec:.0f}초)")
    print(f"장소 {len(PLACES)}개 (저조도 {dark}개) · 재방문 {len(REVISITS)}개")
    print(f"정답: {truth_path}")
    return video_path


def main() -> None:
    parser = argparse.ArgumentParser(description="장소 보존률 검증용 장면 생성")
    parser.add_argument("--out-dir", type=Path, default=Path("scene_out"))
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seconds-per-segment", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    render(
        args.out_dir,
        fps=args.fps,
        seconds_per_segment=args.seconds_per_segment,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
