"""간판·문자 영역 saliency.

Pind의 장소 추출 정확도는 사실상 "간판을 읽었는가"에 달려 있다(`visual_analyzer` 프롬프트가
signage를 CRUCIAL로 명시한 것과 같은 이유). 그래서 프레임을 고를 때 "문자가 있을 가능성"을
직접 점수화해 같은 샷 안에서도 간판이 크게 잡힌 프레임을 대표로 올린다.

방식은 MSER(Maximally Stable Extremal Regions) + 기하 필터다. Neumann & Matas,
"Real-Time Scene Text Localization and Recognition" (CVPR 2012)의 전처리 단계를 단순화한 것으로,
검출기 가중치를 받지 않아 CPU에서 프레임당 수 ms 수준으로 끝난다. 목적이 인식이 아니라
"이 프레임을 VLM에 올릴 가치가 있는가"의 정렬이므로 이 정밀도로 충분하다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]


@dataclass(frozen=True)
class TextnessResult:
    """문자 영역 추정 결과."""

    score: float
    """0~1 점수. 면적 점유율에 정렬 보너스를 더한 값."""

    region_count: int
    """기하 필터를 통과한 문자 후보 영역 수."""

    area_ratio: float
    """후보 영역이 프레임에서 차지하는 면적 비율."""

    boxes: tuple[tuple[int, int, int, int], ...]
    """(x, y, w, h) 후보 박스. 디버그 시각화용."""


def _passes_geometry(width: int, height: int, frame_area: float) -> bool:
    """문자 획(stroke)으로 보기 어려운 영역을 떨군다."""
    if height == 0 or width == 0:
        return False
    aspect = width / height
    area = width * height
    return (
        0.08 <= aspect <= 12.0  # 세로 간판(한글 세로쓰기)까지 허용
        and 8 <= height <= 400
        and area >= 40
        and area / frame_area <= 0.20  # 벽면 전체처럼 큰 균일 영역 제외
    )


def _row_alignment_bonus(boxes: list[tuple[int, int, int, int]], frame_height: int) -> float:
    """같은 줄에 정렬된 영역이 많으면 문자열일 가능성이 높다는 사전지식을 점수화."""
    if len(boxes) < 2 or frame_height == 0:
        return 0.0
    band = max(frame_height // 40, 4)
    rows: dict[int, int] = {}
    for _, y, _, height in boxes:
        key = (y + height // 2) // band
        rows[key] = rows.get(key, 0) + 1
    largest_row = max(rows.values())
    return float(np.clip((largest_row - 1) / 8.0, 0.0, 1.0))


def textness(image: BgrImage, *, delta: int = 5, max_regions: int = 2000) -> TextnessResult:
    """프레임의 문자 영역 점수를 계산한다.

    Args:
        image: BGR uint8 프레임.
        delta: MSER 안정성 델타. 작을수록 더 많은 영역을 잡는다.
        max_regions: 처리 상한. 텍스처가 심한 프레임에서 시간 폭주를 막는다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    frame_area = float(height * width)
    if frame_area == 0:
        return TextnessResult(score=0.0, region_count=0, area_ratio=0.0, boxes=())

    mser = cv2.MSER.create()
    mser.setDelta(delta)
    mser.setMinArea(40)
    mser.setMaxArea(int(frame_area * 0.20))
    regions, _ = mser.detectRegions(gray)

    boxes: list[tuple[int, int, int, int]] = []
    for region in regions[:max_regions]:
        points = np.asarray(region, dtype=np.int32).reshape(-1, 1, 2)
        box_x, box_y, box_w, box_h = cv2.boundingRect(points)
        if _passes_geometry(box_w, box_h, frame_area):
            boxes.append((int(box_x), int(box_y), int(box_w), int(box_h)))

    if not boxes:
        return TextnessResult(score=0.0, region_count=0, area_ratio=0.0, boxes=())

    # 박스가 겹치므로 면적을 단순 합산하면 과대평가된다. 마스크로 합집합 면적을 구한다.
    mask = np.zeros((height, width), dtype=np.uint8)
    for box_x, box_y, box_w, box_h in boxes:
        mask[box_y : box_y + box_h, box_x : box_x + box_w] = 1
    area_ratio = float(mask.sum()) / frame_area

    # 정규화 상수는 "실사 프레임에서 이 정도면 간판이 큼직하게 잡힌 것"을 기준으로 잡았다.
    # 합성 테스트 영상처럼 인공 도형이 많은 입력에서는 쉽게 포화하므로, 실제 영상으로
    # 재튜닝할 여지가 있는 값이다.
    coverage = float(np.clip(area_ratio / 0.35, 0.0, 1.0))
    density = float(np.clip(len(boxes) / 120.0, 0.0, 1.0))
    alignment = _row_alignment_bonus(boxes, height)
    score = float(np.clip(0.5 * coverage + 0.2 * density + 0.3 * alignment, 0.0, 1.0))

    return TextnessResult(
        score=score,
        region_count=len(boxes),
        area_ratio=area_ratio,
        boxes=tuple(boxes),
    )
