"""학습된 장면 문자 검출기 — 프레임 선별의 "간판이 보이는가" 신호.

## 왜 MSER textness 를 대체하는가

`textness.py`(MSER + 기하 필터)는 합성 장면에서는 문자 위치를 잘 짚었지만, 유튜브 실영상
3편(1,876 프레임)에서는 **85–92% 프레임에서 점수가 1.0으로 포화**했다. 벽돌·나뭇잎·메뉴판처럼
텍스처가 많은 실사 프레임에서는 MSER 영역이 중앙값 1,000개를 넘기 때문이다. "이 프레임에서
장소 문자열이 읽히는가"를 가르는 AUC가 0.46–0.56이었고, 그 점수로 고른 16장은 무작위보다
나쁘거나 비슷했다 (`docs/vision-frontend.md` 「유튜브 실측 결과」).

같은 프레임에서 PP-OCRv4 검출기(DB, Differentiable Binarization)의 **검출 박스 수**는 AUC
0.59–0.84였고, 그 순서로 고르면 16장에 18곳 중 8곳(무작위 기댓값 2.7)을 남겼다. 다만 이 선택은
사전 등록 테스트(새 영상 3편)에서 재현되지 않아 기본값이 되지 못했다
(`docs/vision-frontend-decisions.md` #15–16). 검출기 자체는 원본과 일치하므로 다른 선택 방식의
입력으로 재사용할 수 있다.

## 구현

RapidOCR(`rapidocr_onnxruntime` 1.4.4)의 검출 단계만 떼어 **onnxruntime + OpenCV** 로 다시 썼다.
그 패키지는 Python 3.13 을 지원하지 않고(`<3.13`) 인식·방향 분류 모델까지 끌어오기 때문이다.
전처리(짧은 변 736, 32 배수, mean/std 0.5)와 후처리(확률 0.3 이진화 → 2×2 팽창 → 윤곽 →
최소 외접 사각형 → 평균 확률 0.5 이상, 짧은 변 3 이상)는 원본과 같다. 원본의 `unclip`
(pyclipper 다각형 확장)은 박스 **수**를 바꾸지 않으므로, 면적 계산에만 사각형 근사
(각 변 +2d, d = 면적×1.6/둘레)로 대신한다. 원본과의 일치는 유튜브 1,876 프레임으로 확인했다.

모델은 레포에 넣지 않는다. PyPI 의 RapidOCR 휠(Apache-2.0)에서 검출 모델 한 파일만 꺼내
SHA-256 을 확인해 캐시한다:

    python -m app.pipeline.vision.textdet --fetch
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import structlog
from numpy.typing import NDArray

logger = structlog.get_logger(__name__)

BgrImage = NDArray[np.uint8]

MODEL_FILENAME = "ch_PP-OCRv4_det_infer.onnx"
MODEL_SHA256 = "d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/ba/12/"
    "1e5497183bdbe782dbb91bad1d0d2297dba4d2831b2652657f7517bfc6df/"
    "rapidocr_onnxruntime-1.4.4-py3-none-any.whl"
)
WHEEL_SHA256 = "971d7d5f223a7a808662229df1ef69893809d8457d834e6373d3854bc1782cbf"
WHEEL_MEMBER = f"rapidocr_onnxruntime/models/{MODEL_FILENAME}"

# RapidOCR config.yaml 의 Det 값 그대로
LIMIT_SIDE_LEN = 736
PROB_THRESH = 0.3
BOX_THRESH = 0.5
MAX_CANDIDATES = 1000
UNCLIP_RATIO = 1.6
MIN_SIZE = 3


class TextDetectorUnavailable(RuntimeError):
    """onnxruntime 또는 모델 파일이 없음."""


@dataclass(frozen=True)
class TextDetection:
    """프레임 한 장의 문자 검출 결과."""

    box_count: int
    """검출된 문자열 박스 수. 프레임 선별 점수로 쓴다."""

    area_ratio: float
    """박스(확장 근사) 면적 합 / 프레임 면적. 겹침은 합산된다."""


def default_model_path() -> Path:
    override = os.environ.get("PIND_TEXTDET_MODEL")
    if override:
        return Path(override)
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "pind" / MODEL_FILENAME


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_model(target: Path | None = None) -> Path:
    """PyPI 휠에서 검출 모델을 꺼내 저장한다. 이미 있고 해시가 맞으면 그대로 둔다."""
    path = target or default_model_path()
    if path.exists() and _sha256(path.read_bytes()) == MODEL_SHA256:
        return path
    with urllib.request.urlopen(WHEEL_URL, timeout=120) as response:  # noqa: S310 - 고정 https URL
        wheel = response.read()
    if _sha256(wheel) != WHEEL_SHA256:
        raise TextDetectorUnavailable("RapidOCR 휠 해시가 다릅니다 — 다운로드가 손상되었습니다")
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        model = archive.read(WHEEL_MEMBER)
    if _sha256(model) != MODEL_SHA256:
        raise TextDetectorUnavailable("검출 모델 해시가 다릅니다")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model)
    return path


def _preprocess(image: BgrImage) -> NDArray[np.float32]:
    height, width = image.shape[:2]
    ratio = LIMIT_SIDE_LEN / min(height, width) if min(height, width) < LIMIT_SIDE_LEN else 1.0
    resize_h = max(int(round(int(height * ratio) / 32) * 32), 32)
    resize_w = max(int(round(int(width * ratio) / 32) * 32), 32)
    resized = cv2.resize(image, (resize_w, resize_h))
    normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)


def _box_score(prob: NDArray[np.float32], points: NDArray[np.float32]) -> float:
    """최소 외접 사각형 안의 평균 확률 (RapidOCR `box_score_fast`)."""
    h, w = prob.shape
    xmin = int(np.clip(np.floor(points[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(points[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(points[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(points[:, 1].max()), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    shifted = points - np.array([xmin, ymin], dtype=np.float32)
    cv2.fillPoly(mask, [shifted.reshape(-1, 2).astype(np.int32)], 1)
    return float(cv2.mean(prob[ymin : ymax + 1, xmin : xmax + 1], mask)[0])


def postprocess(prob: NDArray[np.float32], image_shape: tuple[int, int]) -> TextDetection:
    """확률 맵(H', W') → 원본 좌표계 박스 수·면적."""
    src_h, src_w = image_shape
    map_h, map_w = prob.shape
    bitmap = cv2.dilate((prob > PROB_THRESH).astype(np.uint8), np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    scale_x, scale_y = src_w / map_w, src_h / map_h

    count = 0
    area = 0.0
    for contour in contours[:MAX_CANDIDATES]:
        (_, _), (rect_w, rect_h), _ = rect = cv2.minAreaRect(contour)
        if min(rect_w, rect_h) < MIN_SIZE:
            continue
        points = cv2.boxPoints(rect).astype(np.float32)
        if _box_score(prob, points) < BOX_THRESH:
            continue
        # unclip 근사: 사각형 각 변을 d 만큼 바깥으로
        distance = (rect_w * rect_h) * UNCLIP_RATIO / (2.0 * (rect_w + rect_h))
        grown_w, grown_h = rect_w + 2 * distance, rect_h + 2 * distance
        if min(grown_w, grown_h) < MIN_SIZE + 2:
            continue
        # 원본 좌표계로 옮긴 뒤 3px 이하 박스 제거 (RapidOCR `filter_tag_det_res`)
        angle = np.deg2rad(rect[2])
        out_w = grown_w * np.hypot(np.cos(angle) * scale_x, np.sin(angle) * scale_y)
        out_h = grown_h * np.hypot(np.sin(angle) * scale_x, np.cos(angle) * scale_y)
        if int(out_w) <= 3 or int(out_h) <= 3:
            continue
        count += 1
        area += float(out_w * out_h)
    return TextDetection(box_count=count, area_ratio=area / float(src_h * src_w))


class TextDetector(Protocol):
    """프레임 선별이 요구하는 검출기 인터페이스 (테스트에서 가짜로 바꿔 끼운다)."""

    name: str

    def detect(self, image: BgrImage) -> TextDetection: ...


class DbTextDetector:
    """PP-OCRv4 DB 검출기 (onnxruntime, CPU)."""

    name = "ppocr-v4-det"

    def __init__(self, model_path: Path | None = None, *, threads: int = 0) -> None:
        path = model_path or default_model_path()
        if not path.exists():
            raise TextDetectorUnavailable(
                f"검출 모델이 없습니다: {path}\n  python -m app.pipeline.vision.textdet --fetch"
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise TextDetectorUnavailable(
                "onnxruntime 이 필요합니다: pip install -e '.[vision-textdet]'"
            ) from exc
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self._session: Any = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def detect(self, image: BgrImage) -> TextDetection:
        prob = self._session.run(None, {self._input: _preprocess(image)})[0][0, 0]
        return postprocess(prob.astype(np.float32), (image.shape[0], image.shape[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="장면 문자 검출 모델")
    parser.add_argument("--fetch", action="store_true", help="모델을 받아 캐시에 저장")
    parser.add_argument("--path", type=Path, default=None)
    args = parser.parse_args()
    if args.fetch:
        print(fetch_model(args.path))


if __name__ == "__main__":
    main()
