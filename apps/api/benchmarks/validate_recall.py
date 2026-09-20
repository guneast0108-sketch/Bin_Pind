"""장소 보존률(recall) 검증 — "얼마나 줄였나"가 아니라 "무엇을 잃었나"를 잰다.

## 왜 감소율로는 부족한가

`bench_keyframes.py`는 프레임을 몇 %로 줄였는지 보여준다. 그 숫자는 혼자서는 의미가
없다. 프레임을 1장만 남기면 감소율은 99%가 되고 장소는 거의 다 사라진다. 이 파이프라인이
지켜야 하는 것은 **줄이면서도 장소를 잃지 않는 것**이므로, 재야 하는 값은 감소율이 아니라
보존률이다.

## 판정자를 파이프라인 밖에 둔다

"이 프레임에서 간판을 읽을 수 있는가"를 파이프라인 자신의 `textness` 점수로 판정하면
동어반복이 된다(선별 점수에 이미 textness가 들어간다). 그래서 판정은 **외부 OCR
(tesseract)**이 한다. 파이프라인이 고른 프레임을 OCR에 넣어 정답 문자열이 읽히는지
보고, 읽히면 그 장소는 살아남은 것으로 센다.

OCR은 같은 이미지에서도 `--psm` 모드에 따라 결과가 갈린다. 그래서 여러 모드의 출력을
합집합으로 쓰고, 문자열 비교는 정확 일치가 아니라 유사도 임계(기본 0.80)로 한다
(`CAFE LUMIERE`를 `care Lumiere`로 읽는 경우를 사람은 맞다고 보기 때문이다). 한글은
자모 단위로 비교한다(`recall_truth.py` 참고). 같은 기준을 모든 설정에 **동일하게**
적용하므로 비교는 공정하다.

## 지표

- **장소 보존률** — 정답 장소 중 선별 프레임에서 정답 문자열이 읽힌 비율. 저조도/정상
  노출로 나눠서 본다. 저조도 쪽이 미니 ISP가 실제로 기여하는지를 보여준다.
- **간판 보존률** — 그중 **간판(sign)으로** 읽힌 비율. 유튜브 브이로그는 상호를 편집 자막에
  넣는 경우가 많다. 자막은 화질 처리 없이도 읽히므로, 자막으로 읽힌 장소까지 섞으면
  보정의 기여가 가려진다.
- **오라클 보존률** — 선별 없이 모든 샘플 프레임을 OCR했을 때의 보존률. 상한선이다.
  이 값과의 차이가 "선별 때문에 잃은 것"이다.
- **장소당 프레임 수** — 1에 가까울수록 중복이 잘 접혔다.

## 사용

    # 정답을 아는 합성 장면으로
    python benchmarks/make_recall_scene.py --out-dir scene_out
    python benchmarks/validate_recall.py --scene-dir scene_out

    # 유튜브 영상으로 (매니페스트 + 사람이 채운 정답 JSON)
    python benchmarks/youtube_fetch.py
    python benchmarks/validate_recall.py --manifest benchmarks/youtube/manifest.json

    # 영상 파일 하나 + 정답 JSON
    python benchmarks/validate_recall.py --video trip.mp4 --ground-truth truth.json

정답 JSON 형식은 `recall_truth.py` 모듈 설명 참고.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
from numpy import ndarray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline.vision import (  # noqa: E402
    DbTextDetector,
    Embedder,
    KeyframeSelection,
    RejectReason,
    TextDetectorUnavailable,
    VisionFrontendConfig,
    build_embedder,
    select_keyframes,
)
from app.pipeline.vision.decode import probe_duration_sec, read_bgr  # noqa: E402
from recall_truth import (  # noqa: E402
    PlaceTruth,
    TextSource,
    TruthError,
    format_time,
    load_manifest,
    load_truth,
    matched_sources,
    select_specs,
)

# 여러 psm 모드의 출력을 합집합으로 쓴다. 한 모드만 쓰면 사람이 읽을 수 있는 간판도
# 모드에 따라 통째로 놓친다(psm 3은 큰 글자에, psm 6은 블록 텍스트에 강하다).
OCR_PSM_MODES: tuple[str, ...] = ("3", "6", "11")

# 문자열 유사도 임계. OCR 오인식(CAFE→care)을 사람 기준으로 흡수한다.
MATCH_THRESHOLD = 0.80

# "dark": "auto" 판정 기준 — 구간 프레임 평균 휘도(0~1)의 중앙값이 이보다 낮으면 저조도.
# 파이프라인의 저조도 구제 기준(rescue_luma_below)과 같은 값을 쓴다.
DARK_LUMA_BELOW = 0.18


# --------------------------------------------------------------------------- OCR


JUDGES: tuple[str, ...] = ("tesseract", "easyocr")

# tesseract 언어 코드 → EasyOCR 언어 코드
_EASYOCR_LANG = {"kor": "ko", "eng": "en", "jpn": "ja", "chi_sim": "ch_sim"}


@cache
def _installed_languages() -> frozenset[str]:
    if shutil.which("tesseract") is None:
        raise SystemExit(
            "tesseract 가 필요합니다.\n"
            "  macOS: brew install tesseract tesseract-lang\n"
            "  Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-kor"
        )
    listed = subprocess.run(  # noqa: S603 - 고정 인자
        ["tesseract", "--list-langs"], capture_output=True, text=True, check=False
    )
    lines = (listed.stdout + listed.stderr).splitlines()
    return frozenset(line.strip() for line in lines[1:] if line.strip())


def require_judge(judge: str, lang: str) -> None:
    """판정자 실행 환경을 미리 확인한다 — 오라클 도중에 실패하면 수십 분이 날아간다."""
    if judge not in JUDGES:
        raise SystemExit(f"알 수 없는 판정자: {judge} (가능: {JUDGES})")
    if judge == "easyocr":
        try:
            import easyocr  # noqa: F401
        except ImportError as exc:
            raise SystemExit("EasyOCR 가 필요합니다: pip install -e '.[bench]'") from exc
        unknown = [code for code in lang.split("+") if code not in _EASYOCR_LANG]
        if unknown:
            raise SystemExit(f"EasyOCR 언어 매핑이 없습니다: {unknown}")
        return
    missing = [code for code in lang.split("+") if code not in _installed_languages()]
    if missing:
        raise SystemExit(
            f"tesseract 언어 데이터가 없습니다: {missing}\n"
            "  macOS: brew install tesseract-lang\n"
            "  Ubuntu: sudo apt install tesseract-ocr-kor"
        )


@cache
def _easyocr_reader(languages: tuple[str, ...]) -> Any:
    import easyocr

    return easyocr.Reader(list(languages), gpu=False, verbose=False)


@dataclass
class OcrEngine:
    """외부 판정자(OCR) 호출 + 디스크 캐시.

    판정자는 두 가지다.

    - `tesseract` — 문서 OCR. 합성 장면(단색 판 위 영문 간판)에서는 충분하고, 기존 수치가
      이 판정자로 나왔으므로 합성 장면의 기본값으로 유지한다.
    - `easyocr` — 장면 문자(scene text) 검출(CRAFT) + 인식. 유튜브 영상의 기본값이다.
      한글 간판을 배경 위에 얹은 스모크 영상에서 tesseract 는 크고 선명한 `용강식당` 도
      못 읽었고(psm 3/6/11 모두), EasyOCR 은 8장 모두에서 문자열을 찾았다. 판정자가 약하면
      오라클이 낮아져 **파이프라인이 무엇을 잃었는지 구분할 수 없다.**

    ablation 4종은 같은 원본 프레임을 여러 번 OCR하고, 오라클은 모든 샘플 프레임을 OCR한다.
    한글 모델은 느려서 캐시 없이는 영상 한 편에 수십 분이 걸린다. 키는 이미지 바이트 +
    판정자 + 언어라 보정된 프레임과 원본 프레임은 따로 캐시된다.
    """

    lang: str = "eng"
    judge: str = "tesseract"
    cache_dir: Path | None = None
    calls: int = 0
    hits: int = 0

    @property
    def label(self) -> str:
        if self.judge == "tesseract":
            return f"tesseract `-l {self.lang}`, psm {'/'.join(OCR_PSM_MODES)} 합집합"
        return f"EasyOCR `{self.lang}`"

    def read(self, image: ndarray) -> str:
        key = hashlib.sha1(
            image.tobytes()
            + repr(image.shape).encode()
            + self.judge.encode()
            + self.lang.encode()
            + ",".join(OCR_PSM_MODES).encode()
        ).hexdigest()
        cached = self.cache_dir / f"{key}.txt" if self.cache_dir is not None else None
        if cached is not None and cached.exists():
            self.hits += 1
            return cached.read_text(encoding="utf-8")

        self.calls += 1
        text = self._easyocr(image) if self.judge == "easyocr" else self._tesseract(image)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(text, encoding="utf-8")
        return text

    def _tesseract(self, image: ndarray) -> str:
        with tempfile.TemporaryDirectory() as work:
            image_path = Path(work) / "frame.png"
            cv2.imwrite(str(image_path), image)
            chunks: list[str] = []
            for psm in OCR_PSM_MODES:
                completed = subprocess.run(  # noqa: S603 - 고정 인자, 셸 없음
                    ["tesseract", str(image_path), "stdout", "-l", self.lang, "--psm", psm],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                chunks.append(completed.stdout)
        return " ".join(chunks)

    def _easyocr(self, image: ndarray) -> str:
        languages = tuple(_EASYOCR_LANG[code] for code in self.lang.split("+"))
        detections = _easyocr_reader(languages).readtext(image, detail=0, paragraph=False)
        # 신뢰도로 거르지 않는다 — 맞았는지는 정답 문자열과의 유사도가 정한다.
        return " ".join(str(text) for text in detections)


# --------------------------------------------------------------------------- 평가


@dataclass
class RunResult:
    """설정 하나로 돌린 결과."""

    label: str
    selected_frames: int
    decoded_frames: int
    recovered: set[str]
    recovered_by_sign: set[str]
    frames_per_place: dict[str, int]
    loss_reason: dict[str, str]
    elapsed_sec: float


@dataclass
class OracleResult:
    """선별 없이 모든 샘플 프레임을 본 결과 + 구간 휘도."""

    recovered: set[str] = field(default_factory=set)
    recovered_by_sign: set[str] = field(default_factory=set)
    luma_by_place: dict[str, list[float]] = field(default_factory=dict)
    sampled_frames: int = 0
    # --full-table 일 때만: 샘플 인덱스 → 그 프레임에서 읽힌 장소 (전체 / 간판)
    frame_table: dict[int, set[str]] | None = None
    frame_table_sign: dict[int, set[str]] | None = None
    # 무작위 K장 기댓값: 장소별 "한 번이라도 뽑힐 확률" (전체 / 간판)
    random_hit: dict[str, float] | None = None
    random_hit_sign: dict[str, float] | None = None


def _place_of(places: list[PlaceTruth], timestamp_sec: float) -> PlaceTruth | None:
    for place in places:
        if place.contains(timestamp_sec):
            return place
    return None


def _needs_more(place: PlaceTruth, recovered: set[str], by_sign: set[str]) -> bool:
    """이 장소를 더 볼 필요가 있나 — 간판 문자열이 있으면 간판으로 읽힐 때까지 본다."""
    if place.place_id not in recovered:
        return True
    return place.has_sign and place.place_id not in by_sign


def _record(
    place: PlaceTruth,
    sources: set[TextSource],
    recovered: set[str],
    by_sign: set[str],
) -> None:
    if sources:
        recovered.add(place.place_id)
    if "sign" in sources:
        by_sign.add(place.place_id)


def evaluate(
    selection: KeyframeSelection,
    places: list[PlaceTruth],
    *,
    label: str,
    threshold: float,
    ocr: OcrEngine,
) -> RunResult:
    """선별 결과를 정답과 맞춰 보존률과 **손실 원인**을 낸다.

    보존률만 내면 "왜 잃었는지"를 알 수 없어 고칠 수가 없다. 장소를 잃는 경로는
    세 가지뿐이고, 각각 고쳐야 할 곳이 다르다.

    - 게이트 탈락 — 화질 게이트가 그 장소의 프레임을 다 버렸다 (임계/구제 문제)
    - 선별 제외 — 게이트는 통과했는데 대표로 뽑히지 않았다 (샷 분할/점수/예산 문제)
    - 읽기 실패 — 뽑혔는데 OCR이 문자열을 못 읽었다 (보정 문제)
    """
    recovered: set[str] = set()
    by_sign: set[str] = set()
    frames_per_place: dict[str, int] = {place.place_id: 0 for place in places}
    loss_reason: dict[str, str] = {}

    for keyframe in selection.keyframes:
        place = _place_of(places, keyframe.timestamp_sec)
        if place is None:
            continue
        frames_per_place[place.place_id] += 1
        if not _needs_more(place, recovered, by_sign):
            continue
        sources = matched_sources(
            ocr.read(read_bgr(keyframe.path)), place.texts, threshold=threshold
        )
        _record(place, sources, recovered, by_sign)

    rejected_at = {frame.timestamp_sec for frame in selection.rejected}
    sampled_at = {frame.timestamp_sec for frame in selection.frame_stats}

    for place in places:
        if place.place_id in recovered:
            continue
        sampled = {ts for ts in sampled_at if place.contains(ts)}
        survived = sampled - rejected_at
        if not sampled:
            loss_reason[place.place_id] = "샘플 없음"
        elif not survived:
            loss_reason[place.place_id] = "게이트 탈락"
        elif frames_per_place[place.place_id] == 0:
            loss_reason[place.place_id] = "선별 제외"
        else:
            loss_reason[place.place_id] = "읽기 실패"

    return RunResult(
        label=label,
        selected_frames=selection.summary.selected_frames,
        decoded_frames=selection.summary.decoded_frames,
        recovered=recovered,
        recovered_by_sign=by_sign,
        frames_per_place=frames_per_place,
        loss_reason=loss_reason,
        elapsed_sec=selection.summary.elapsed_sec,
    )


def scan_oracle(
    video_path: Path,
    places: list[PlaceTruth],
    *,
    sample_fps: float,
    threshold: float,
    ocr: OcrEngine,
    exhaustive: bool = False,
) -> OracleResult:
    """선별 없이 모든 샘플 프레임을 OCR한 상한선 + 장소 구간의 휘도.

    `exhaustive=True` 이면 장소가 이미 읽혔어도 구간 안 프레임을 전부 OCR해 프레임 단위 표를
    남긴다. 무작위 선별 기댓값처럼 "어떤 프레임 집합을 골랐다면"을 OCR 재실행 없이 계산할 때 쓴다.

    파이프라인이 고른 프레임의 보존률은 이 값과 비교해야 의미가 있다. 오라클이 놓친
    장소는 애초에 영상에서 읽을 수 없는 장소이고, 파이프라인 탓이 아니다.

    프레임은 순차 디코딩(grab)으로 읽는다. 샘플마다 탐색(seek)하면 H.264 유튜브 영상에서는
    매번 앞 키프레임부터 다시 디코딩해 훨씬 느리다.
    """
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / sample_fps)), 1)

    result = OracleResult(luma_by_place={place.place_id: [] for place in places})
    if exhaustive:
        result.frame_table, result.frame_table_sign = {}, {}
    index = 0
    while capture.grab():
        if index % step == 0:
            sample_index = result.sampled_frames
            result.sampled_frames += 1
            place = _place_of(places, index / fps)
            if place is not None:
                ok, frame = capture.retrieve()
                if ok:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    result.luma_by_place[place.place_id].append(float(np.mean(gray)) / 255.0)
                    if exhaustive or _needs_more(place, result.recovered, result.recovered_by_sign):
                        sources = matched_sources(ocr.read(frame), place.texts, threshold=threshold)
                        _record(place, sources, result.recovered, result.recovered_by_sign)
                        if result.frame_table is not None and result.frame_table_sign is not None:
                            if sources:
                                result.frame_table[sample_index] = {place.place_id}
                            if "sign" in sources:
                                result.frame_table_sign[sample_index] = {place.place_id}
        index += 1
    capture.release()
    return result


RANDOM_DRAWS = 2000

_GATE_REASONS = {RejectReason.BLUR, RejectReason.UNDEREXPOSED, RejectReason.OVEREXPOSED}


@cache
def _text_detector() -> DbTextDetector:
    try:
        return DbTextDetector()
    except TextDetectorUnavailable as exc:
        raise SystemExit(str(exc)) from exc


def random_expectation(
    table: dict[int, set[str]],
    allowed: list[int],
    budget: int,
    place_ids: list[str],
    *,
    seed: int = 20260915,
) -> dict[str, float]:
    """게이트 통과 프레임에서 `budget` 장을 무작위로 뽑았을 때 장소별 적중 확률 (몬테카를로).

    "선별기가 무작위보다 나은가"를 영상마다 같은 조건(같은 게이트, 같은 예산)으로 비교하려는
    기준선이다. 장소 수 기댓값은 이 확률의 합이다.
    """
    rng = np.random.default_rng(seed)
    hits = dict.fromkeys(place_ids, 0)
    if not allowed:
        return dict.fromkeys(place_ids, 0.0)
    k = min(budget, len(allowed))
    pool = np.asarray(allowed)
    for _ in range(RANDOM_DRAWS):
        found: set[str] = set()
        for sample in rng.choice(pool, size=k, replace=False):
            found |= table.get(int(sample), set())
        for pid in found:
            hits[pid] += 1
    return {pid: hits[pid] / RANDOM_DRAWS for pid in place_ids}


def resolve_dark(places: list[PlaceTruth], oracle: OracleResult) -> dict[str, bool]:
    """`dark: "auto"` 장소를 구간 휘도 중앙값으로 판정한다. 명시값은 그대로 쓴다."""
    resolved: dict[str, bool] = {}
    for place in places:
        if place.dark is not None:
            resolved[place.place_id] = place.dark
            continue
        lumas = oracle.luma_by_place.get(place.place_id) or []
        resolved[place.place_id] = bool(lumas) and median(lumas) < DARK_LUMA_BELOW
    return resolved


ABLATIONS: dict[str, dict[str, object]] = {
    # 전체 파이프라인 — 기준선
    "full": {},
    # 미니 ISP를 끈다. 저조도 장소가 살아남는지가 여기서 갈린다.
    "no-enhance": {"enhance": False},
    # 저조도 구제를 끈다 = 화질 게이트가 어두운 프레임을 먼저 버리는 예전 순서 재현.
    "no-rescue": {"rescue_luma_below": 0.0, "rescue_clip_low_above": 1.0},
    # 학습된 문자 검출기 박스 수 순 + 시간 간격 (selection_strategy="text_nms").
    "text-det": {"selection_strategy": "text_nms"},
    # 중복 제거를 끈다. 프레임 수가 몇 배로 늘어나는지 = dedup이 버는 비용.
    "no-dedup": {
        "scene_similarity_threshold": 1.0,
        "duplicate_shot_threshold": 1.0,
        "max_keyframes": 200,
    },
}


# --------------------------------------------------------------------------- 보고서


def _ratio(recovered: set[str], subset: set[str]) -> str:
    if not subset:
        return "—"
    return f"{len(recovered & subset)}/{len(subset)}"


def _expected(hit: dict[str, float], subset: set[str]) -> str:
    if not subset:
        return "—"
    return f"{sum(hit[pid] for pid in subset):.1f}/{len(subset)}"


def format_table(
    results: list[RunResult],
    places: list[PlaceTruth],
    oracle: OracleResult,
    dark: dict[str, bool],
    *,
    budget: int = 0,
) -> str:
    all_ids = {place.place_id for place in places}
    dark_ids = {pid for pid in all_ids if dark[pid]}
    bright_ids = all_ids - dark_ids
    sign_ids = {place.place_id for place in places if place.has_sign}
    show_sign = sign_ids != all_ids or any(
        entry.source != "sign" for place in places for entry in place.texts
    )

    sign_header = " 간판으로 |" if show_sign else ""
    sign_rule = "---:|" if show_sign else ""
    lines = [
        f"| 설정 | 선별 프레임 | 장소 보존 |{sign_header} 정상 노출 | 저조도 "
        "| 장소당 프레임 | 소요 |",
        f"|---|---:|---:|{sign_rule}---:|---:|---:|---:|",
    ]
    oracle_sign = f" {_ratio(oracle.recovered_by_sign, sign_ids)} |" if show_sign else ""
    lines.append(
        f"| _오라클 (선별 없음)_ | {oracle.sampled_frames} | "
        f"{_ratio(oracle.recovered, all_ids)} |{oracle_sign} "
        f"{_ratio(oracle.recovered, bright_ids)} | {_ratio(oracle.recovered, dark_ids)} | — | — |"
    )
    if oracle.random_hit is not None and oracle.random_hit_sign is not None:
        hit, hit_sign = oracle.random_hit, oracle.random_hit_sign
        sign_cell = f" {_expected(hit_sign, sign_ids)} |" if show_sign else ""
        lines.append(
            f"| _무작위 (기댓값)_ | {budget} | "
            f"{_expected(hit, all_ids)} |{sign_cell} "
            f"{_expected(hit, bright_ids)} | {_expected(hit, dark_ids)} | — | — |"
        )
    for result in results:
        counted = [n for n in result.frames_per_place.values() if n > 0]
        per_place = sum(counted) / len(counted) if counted else 0.0
        sign_cell = f" {_ratio(result.recovered_by_sign, sign_ids)} |" if show_sign else ""
        lines.append(
            f"| {result.label} | {result.selected_frames} | "
            f"{_ratio(result.recovered, all_ids)} |{sign_cell} "
            f"{_ratio(result.recovered, bright_ids)} | {_ratio(result.recovered, dark_ids)} | "
            f"{per_place:.1f} | {result.elapsed_sec:.1f}s |"
        )
    return "\n".join(lines)


def format_losses(
    results: list[RunResult],
    places: list[PlaceTruth],
    oracle: OracleResult,
    dark: dict[str, bool],
    *,
    clip_start: float,
) -> str:
    lines = ["| 설정 | 잃은 장소 | 원인 | 원본 시각 |", "|---|---|---|---|"]
    for result in results:
        lost = [place for place in places if place.place_id not in result.recovered]
        if not lost:
            lines.append(f"| {result.label} | 없음 | — | — |")
            continue
        for order, place in enumerate(lost):
            label = result.label if order == 0 else ""
            tags = []
            if dark[place.place_id]:
                tags.append("저조도")
            if place.place_id not in oracle.recovered:
                tags.append("오라클도 못 읽음")
            suffix = f" ({', '.join(tags)})" if tags else ""
            when = ", ".join(
                f"{format_time(a + clip_start)}–{format_time(b + clip_start)}"
                for a, b in place.segments
            )
            lines.append(
                f"| {label} | {place.sign_text}{suffix} | "
                f"{result.loss_reason.get(place.place_id, '—')} | {when} |"
            )
    return "\n".join(lines)


@dataclass
class VideoReport:
    """영상 한 편의 결과 — 여러 편 합산에 쓴다."""

    title: str
    places: list[PlaceTruth]
    dark: dict[str, bool]
    oracle: OracleResult
    results: list[RunResult]
    markdown: str
    max_keyframes: int = 0


def run_video(
    video_path: Path,
    places: list[PlaceTruth],
    *,
    title: str,
    out_dir: Path,
    labels: list[str],
    embedder: Embedder,
    sample_fps: float,
    max_keyframes: int,
    threshold: float,
    ocr: OcrEngine,
    clip_start: float = 0.0,
    preamble: str = "",
    full_table: bool = False,
) -> VideoReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(places) > max_keyframes:
        print(
            f"경고: 장소 {len(places)}곳 > 키프레임 예산 {max_keyframes}장 — 보존률 상한이 "
            f"{max_keyframes}/{len(places)} 입니다. --max-keyframes 를 올리거나 구간을 줄이세요"
        )
    print(f"\n=== {title} · 장소 {len(places)}개 · 임베더 {embedder.name} · 판정 {ocr.label}")

    results: list[RunResult] = []
    gate_rejected_by_label: dict[str, set[int]] = {}
    for label in labels:
        fields: dict[str, object] = {"sample_fps": sample_fps, "max_keyframes": max_keyframes}
        fields.update(ABLATIONS[label])
        config = VisionFrontendConfig(**fields)
        work_dir = out_dir / label
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)

        print(f"[{label}] 실행 중...")
        detector = _text_detector() if config.selection_strategy == "text_nms" else None
        selection = select_keyframes(
            video_path, work_dir, config=config, embedder=embedder, text_detector=detector
        )
        if not gate_rejected_by_label:
            gate_rejected_by_label[label] = {
                frame.frame_index for frame in selection.rejected if frame.reason in _GATE_REASONS
            }
        result = evaluate(selection, places, label=label, threshold=threshold, ocr=ocr)
        print(
            f"  선별 {result.selected_frames}장 / 디코딩 {result.decoded_frames}장 · "
            f"장소 {len(result.recovered)}/{len(places)}"
        )
        results.append(result)

    print("오라클(선별 없음) 계산 중...")
    oracle = scan_oracle(
        video_path,
        places,
        sample_fps=sample_fps,
        threshold=threshold,
        ocr=ocr,
        exhaustive=full_table,
    )
    if oracle.frame_table is not None and oracle.frame_table_sign is not None:
        rejected = next(iter(gate_rejected_by_label.values()), set())
        allowed = [index for index in range(oracle.sampled_frames) if index not in rejected]
        place_ids = [place.place_id for place in places]
        oracle.random_hit = random_expectation(
            oracle.frame_table, allowed, max_keyframes, place_ids
        )
        oracle.random_hit_sign = random_expectation(
            oracle.frame_table_sign, allowed, max_keyframes, place_ids
        )
        table_path = out_dir / "frame_table.json"
        table_path.write_text(
            json.dumps(
                {
                    "sampled_frames": oracle.sampled_frames,
                    "gate_rejected": sorted(rejected),
                    "any": {str(k): sorted(v) for k, v in sorted(oracle.frame_table.items())},
                    "sign": {str(k): sorted(v) for k, v in sorted(oracle.frame_table_sign.items())},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    dark = resolve_dark(places, oracle)

    dark_note = (
        "저조도: 정답 파일 명시값."
        if all(place.dark is not None for place in places)
        else f"저조도: 명시값, 없으면 구간 휘도 중앙값 < {DARK_LUMA_BELOW}."
    )
    sections = [f"## {title}"]
    if preamble:
        sections.append(preamble)
    sections += [
        format_table(results, places, oracle, dark, budget=max_keyframes),
        "### 잃은 장소",
        format_losses(results, places, oracle, dark, clip_start=clip_start),
        f"판정: 외부 OCR({ocr.label}), "
        f"문자열 유사도 {threshold:.2f} 이상(한글은 자모 단위). {dark_note} "
        f"임베더: {embedder.name}. 키프레임 예산: {max_keyframes}장.",
    ]
    markdown = "\n\n".join(sections)
    (out_dir / "recall_report.md").write_text(markdown + "\n", encoding="utf-8")
    print(f"\n{markdown}\n")
    return VideoReport(title, places, dark, oracle, results, markdown, max_keyframes)


def format_summary(reports: list[VideoReport], labels: list[str]) -> str:
    """여러 영상을 합산한 표. 장소 id 는 영상마다 접두사를 붙여 겹치지 않게 한다."""
    lines = [
        "| 설정 | 선별 프레임 | 장소 보존 | 간판으로 | 정상 노출 | 저조도 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def gather(pick: str) -> tuple[int, set[str], set[str]]:
        frames, any_ids, sign_ids = 0, set(), set()
        for number, report in enumerate(reports):
            if pick == "oracle":
                frames += report.oracle.sampled_frames
                rec, by_sign = report.oracle.recovered, report.oracle.recovered_by_sign
            else:
                result = next(r for r in report.results if r.label == pick)
                frames += result.selected_frames
                rec, by_sign = result.recovered, result.recovered_by_sign
            any_ids |= {f"{number}:{pid}" for pid in rec}
            sign_ids |= {f"{number}:{pid}" for pid in by_sign}
        return frames, any_ids, sign_ids

    universe = {f"{n}:{p.place_id}" for n, r in enumerate(reports) for p in r.places}
    dark_u = {f"{n}:{pid}" for n, r in enumerate(reports) for pid, d in r.dark.items() if d}
    sign_u = {f"{n}:{p.place_id}" for n, r in enumerate(reports) for p in r.places if p.has_sign}
    rows = ["oracle", *labels]
    if all(r.oracle.random_hit is not None for r in reports):
        rows.insert(1, "random")
    for pick in rows:
        if pick == "random":
            hit = {
                f"{n}:{pid}": p
                for n, r in enumerate(reports)
                for pid, p in (r.oracle.random_hit or {}).items()
            }
            hit_sign = {
                f"{n}:{pid}": p
                for n, r in enumerate(reports)
                for pid, p in (r.oracle.random_hit_sign or {}).items()
            }
            frames = sum(r.max_keyframes for r in reports)
            lines.append(
                f"| _무작위 (기댓값)_ | {frames} | {_expected(hit, universe)} | "
                f"{_expected(hit_sign, sign_u)} | {_expected(hit, universe - dark_u)} | "
                f"{_expected(hit, dark_u)} |"
            )
            continue
        frames, any_ids, sign_ids = gather(pick)
        name = "_오라클 (선별 없음)_" if pick == "oracle" else pick
        lines.append(
            f"| {name} | {frames} | {_ratio(any_ids, universe)} | {_ratio(sign_ids, sign_u)} | "
            f"{_ratio(any_ids, universe - dark_u)} | {_ratio(any_ids, dark_u)} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- 진입점


def main() -> None:
    parser = argparse.ArgumentParser(description="장소 보존률 검증")
    parser.add_argument("--scene-dir", type=Path, help="make_recall_scene.py 의 출력 폴더")
    parser.add_argument("--manifest", type=Path, help="유튜브 매니페스트 (youtube_fetch.py)")
    parser.add_argument("--only", default="", help="매니페스트 중 이 key 들만 (쉼표 구분)")
    parser.add_argument("--split", default="", help="dev | test 만 (매니페스트)")
    parser.add_argument("--video", type=Path, help="영상 파일")
    parser.add_argument("--ground-truth", type=Path, help="정답 JSON")
    parser.add_argument("--out-dir", type=Path, default=Path("recall_out"))
    parser.add_argument("--embedder", default="auto", help="auto | perceptual | dinov3")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-keyframes", type=int, default=16)
    parser.add_argument("--match-threshold", type=float, default=MATCH_THRESHOLD)
    parser.add_argument(
        "--ocr-lang",
        default=None,
        help="tesseract 언어. 기본: 합성·단일 영상 eng, 매니페스트는 영상별 설정(kor+eng)",
    )
    parser.add_argument(
        "--judge",
        default=None,
        choices=JUDGES,
        help="판정 OCR. 기본: 합성·단일 영상 tesseract, 매니페스트는 영상별 설정(easyocr)",
    )
    parser.add_argument(
        "--full-table",
        action="store_true",
        help="오라클에서 구간 안 프레임을 전부 OCR해 프레임 표와 무작위 선별 기댓값을 낸다 (느림)",
    )
    parser.add_argument("--no-ocr-cache", action="store_true", help="OCR 캐시를 쓰지 않는다")
    parser.add_argument(
        "--ablations",
        default="full,no-enhance,no-rescue,no-dedup",
        help=f"쉼표 구분. 가능: {','.join(ABLATIONS)}",
    )
    args = parser.parse_args()

    labels = [label.strip() for label in args.ablations.split(",") if label.strip()]
    unknown = [label for label in labels if label not in ABLATIONS]
    if unknown:
        raise SystemExit(f"알 수 없는 ablation: {unknown} (가능: {list(ABLATIONS)})")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = build_embedder(args.embedder)
    common = {
        "labels": labels,
        "embedder": embedder,
        "sample_fps": args.sample_fps,
        "max_keyframes": args.max_keyframes,
        "threshold": args.match_threshold,
        "full_table": args.full_table,
    }

    if args.manifest is not None:
        run_manifest(args, out_dir, common)
        return

    if args.scene_dir is not None:
        video_path = args.scene_dir / "scene.mp4"
        truth_path = args.scene_dir / "ground_truth.json"
    elif args.video is not None and args.ground_truth is not None:
        video_path, truth_path = args.video, args.ground_truth
    else:
        raise SystemExit("--scene-dir, --manifest, 또는 (--video 와 --ground-truth)를 지정하세요")
    if not video_path.exists():
        raise SystemExit(f"영상을 찾을 수 없습니다: {video_path}")

    try:
        places, warnings = load_truth(truth_path)
    except TruthError as exc:
        raise SystemExit(f"정답 파일 오류: {exc}") from exc
    for warning in warnings:
        print(f"경고: {warning}")

    lang = args.ocr_lang or "eng"
    judge = args.judge or "tesseract"
    require_judge(judge, lang)
    ocr = OcrEngine(
        lang=lang, judge=judge, cache_dir=None if args.no_ocr_cache else out_dir / ".ocr_cache"
    )
    run_video(
        video_path,
        places,
        title="장소 보존률",
        out_dir=out_dir,
        ocr=ocr,
        **common,
    )
    print(f"리포트: {out_dir / 'recall_report.md'} · OCR {ocr.calls}회 (캐시 {ocr.hits}회)")


def run_manifest(args: argparse.Namespace, out_dir: Path, common: dict[str, object]) -> None:
    manifest_path: Path = args.manifest
    try:
        specs = load_manifest(manifest_path)
    except TruthError as exc:
        raise SystemExit(f"매니페스트 오류: {exc}") from exc
    try:
        specs = select_specs(specs, only=args.only, split=args.split)
    except TruthError as exc:
        raise SystemExit(str(exc)) from exc

    cache_root = manifest_path.parent / ".cache"
    reports: list[VideoReport] = []
    skipped: list[str] = []
    for spec in specs:
        video_path = spec.video_path(cache_root)
        if not video_path.exists():
            skipped.append(
                f"{spec.key}: 영상 없음 → python benchmarks/youtube_fetch.py --only {spec.key}"
            )
            continue
        try:
            places, warnings = load_truth(spec.truth_path, clip=spec.clip)
        except FileNotFoundError:
            skipped.append(f"{spec.key}: 정답 파일 없음 ({spec.truth_path})")
            continue
        except TruthError as exc:
            skipped.append(f"{spec.key}: {exc}")
            continue
        for warning in warnings:
            print(f"경고[{spec.key}]: {warning}")
        decoded_cap = VisionFrontendConfig().max_decoded_frames
        needed = probe_duration_sec(video_path) * float(args.sample_fps)
        if needed > decoded_cap:
            skipped.append(
                f"{spec.key}: 샘플 {needed:.0f}장이 디코딩 상한 {decoded_cap}장을 넘습니다 — "
                "뒷부분이 잘려 장소가 '샘플 없음'으로 집계되므로 manifest 의 clip 을 줄이세요"
            )
            continue

        lang = args.ocr_lang or spec.ocr_lang
        judge = args.judge or spec.judge
        require_judge(judge, lang)
        ocr = OcrEngine(
            lang=lang, judge=judge, cache_dir=None if args.no_ocr_cache else cache_root / "ocr"
        )
        info_path = cache_root / spec.key / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        clip_note = (
            f"{format_time(spec.clip[0])}–{format_time(spec.clip[1])}" if spec.clip else "전체"
        )
        preamble = "\n".join(
            [
                f"- 영상: [{info.get('title', spec.url)}]({spec.url}) · {info.get('channel', '')}",
                f"- 역할: {spec.role or '—'}",
                f"- 구간: {clip_note} · 포맷 {info.get('format_id', '?')} "
                f"({info.get('width', '?')}x{info.get('height', '?')}, "
                f"{info.get('vcodec', '?')}) · sha256 {str(info.get('sha256', '?'))[:12]}",
            ]
        )
        reports.append(
            run_video(
                video_path,
                places,
                title=spec.key,
                out_dir=out_dir / spec.key,
                ocr=ocr,
                clip_start=spec.clip[0] if spec.clip else 0.0,
                preamble=preamble,
                **common,  # type: ignore[arg-type]
            )
        )
        print(f"OCR {ocr.calls}회 (캐시 {ocr.hits}회)")

    if skipped:
        print("\n건너뛴 영상:\n  " + "\n  ".join(skipped))
    if not reports:
        raise SystemExit("평가할 수 있는 영상이 없습니다")

    labels = common["labels"]
    assert isinstance(labels, list)
    summary = "\n\n".join(
        [
            "# 유튜브 영상 장소 보존률",
            f"영상 {len(reports)}편 합산 · 장소 {sum(len(r.places) for r in reports)}곳",
            format_summary(reports, labels),
            *[report.markdown for report in reports],
        ]
    )
    if skipped:
        summary += "\n\n## 건너뛴 영상\n\n" + "\n".join(f"- {line}" for line in skipped)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    print(f"\n합산 리포트: {summary_path}")


if __name__ == "__main__":
    main()
