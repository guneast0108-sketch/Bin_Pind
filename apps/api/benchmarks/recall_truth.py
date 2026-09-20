"""장소 보존률 검증의 정답 형식과 문자열 판정 — OpenCV·tesseract 의존 없는 순수 로직.

`validate_recall.py`(판정 실행)와 `youtube_fetch.py`(유튜브 입력 준비)가 함께 쓰고,
단위 테스트가 이 모듈을 직접 검사한다.

## 정답 JSON (v2)

    {
      "places": [
        {
          "place_id": "yonggang",
          "name": "용강식당",                      # 사람이 보는 이름. 판정에 안 쓴다
          "texts": [                              # 화면에 실제로 보이는 문자열
            {"text": "용강식당", "source": "sign"},
            {"text": "용강식당", "source": "caption"}
          ],
          "dark": "auto",                         # true | false | "auto"
          "segments": [["5:12", "5:40"], [901.0, 915.5]]
        }
      ]
    }

- `segments` 는 **원본 영상 기준 시각**이다(유튜브 플레이어에 보이는 시각 그대로).
  `"m:ss"` · `"h:mm:ss"` 문자열과 초 단위 숫자를 모두 받는다. 클립을 잘라 쓰면 로더가
  클립 시작 시각을 빼서 클립 기준으로 바꾼다.
- `source` 는 그 문자열이 어디에 보이는지다. `sign`(간판·상호 표지), `caption`(편집 자막),
  `other`(메뉴판·영수증 등). 유튜브 브이로그는 자막에 상호를 넣는 경우가 많아서, 자막으로만
  읽힌 장소를 간판으로 읽힌 장소와 섞으면 **화질 처리의 기여를 잴 수 없다**. 보고서는 둘을
  나눠 낸다.
- v1 형식(`"sign_text": "..."`, `"dark": false`)도 그대로 읽는다 — 합성 장면이 쓰는 형식이다.

## 문자열 비교는 자모 단위

기존 정규화(`[^A-Z0-9]` 제거)는 한글을 전부 지웠다. 정답이 `"블루보틀 성수"` 이면 정규화
결과가 빈 문자열이 되어 **모든 장소가 조용히 '읽기 실패'로 집계**됐다. 이제 음절을 NFD로
자모까지 풀어서 비교한다. 음절 단위로 비교하면 받침 하나 틀린 OCR 결과(`식당`→`식단`)가
음절 하나를 통째로 틀린 것과 같은 벌점을 받는다. 영문 `CAFE`→`care` 를 봐주는 것과 같은
기준을 한글에도 적용하려면 자모 단위가 맞다.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

TextSource = Literal["sign", "caption", "other"]
TEXT_SOURCES: tuple[TextSource, ...] = ("sign", "caption", "other")

# 영문 대문자·숫자 + 한글 조합형 자모(U+1100–U+11FF)만 남긴다.
_KEEP = re.compile(r"[^A-Z0-9ᄀ-ᇿ]+")

# 정규화 후 이보다 짧은 문자열은 우연 일치가 쉬워 경고한다(예: "CU" 두 글자).
MIN_RELIABLE_LENGTH = 4


class TruthError(ValueError):
    """정답 파일이 형식에 맞지 않거나 라벨이 덜 채워짐."""


@dataclass(frozen=True)
class PlaceText:
    """장소를 식별하는 화면 문자열 하나."""

    text: str
    source: TextSource

    @property
    def normalized(self) -> str:
        return normalize_text(self.text)


@dataclass(frozen=True)
class PlaceTruth:
    """정답 장소 하나. 구간은 **평가 대상 영상(클립) 기준** 초 단위."""

    place_id: str
    name: str
    texts: tuple[PlaceText, ...]
    dark: bool | None  # None = "auto" — 영상에서 휘도로 판정
    segments: tuple[tuple[float, float], ...]

    def contains(self, timestamp_sec: float) -> bool:
        return any(start <= timestamp_sec <= end for start, end in self.segments)

    @property
    def sign_text(self) -> str:
        """보고서 표기용 대표 문자열 (v1 호환 이름)."""
        return self.name or self.texts[0].text

    @property
    def has_sign(self) -> bool:
        return any(entry.source == "sign" for entry in self.texts)


# --------------------------------------------------------------------------- 시각


def parse_time(value: object) -> float:
    """`12.5` · `"12.5"` · `"5:12"` · `"1:05:12"` → 초."""
    if isinstance(value, bool):
        raise TruthError(f"시각으로 읽을 수 없습니다: {value!r}")
    if isinstance(value, int | float):
        seconds = float(value)
    elif isinstance(value, str):
        parts = value.strip().split(":")
        if not 1 <= len(parts) <= 3 or any(part.strip() == "" for part in parts):
            raise TruthError(f"시각 형식이 아닙니다: {value!r}")
        try:
            numbers = [float(part) for part in parts]
        except ValueError as exc:
            raise TruthError(f"시각 형식이 아닙니다: {value!r}") from exc
        seconds = 0.0
        for number in numbers:
            seconds = seconds * 60.0 + number
    else:
        raise TruthError(f"시각으로 읽을 수 없습니다: {value!r}")
    if seconds < 0:
        raise TruthError(f"음수 시각: {value!r}")
    return seconds


def format_time(seconds: float) -> str:
    """초 → `m:ss` (1시간 이상은 `h:mm:ss`). 유튜브 플레이어 표기와 맞춘다."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# --------------------------------------------------------------------------- 문자열


def normalize_text(text: str) -> str:
    """대소문자·공백·기호를 지우고 한글은 자모로 푼다."""
    folded = unicodedata.normalize("NFKC", text).upper()
    return _KEEP.sub("", unicodedata.normalize("NFD", folded))


def best_similarity(haystack_normalized: str, needle_normalized: str) -> float:
    """needle 과 같은 길이의 창을 haystack 위로 훑은 최대 유사도."""
    needle, haystack = needle_normalized, haystack_normalized
    if not needle or len(haystack) < len(needle):
        return 0.0
    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(needle)
    best = 0.0
    for start in range(len(haystack) - len(needle) + 1):
        matcher.set_seq1(haystack[start : start + len(needle)])
        # real_quick_ratio/quick_ratio 는 ratio 의 상한 — 가망 없는 창을 싸게 건너뛴다.
        if matcher.real_quick_ratio() <= best or matcher.quick_ratio() <= best:
            continue
        best = max(best, matcher.ratio())
        if best == 1.0:
            break
    return best


def text_is_readable(ocr_output: str, text: str, *, threshold: float) -> bool:
    """OCR 결과 안에 정답 문자열이 (오인식을 허용해) 들어 있는지."""
    return best_similarity(normalize_text(ocr_output), normalize_text(text)) >= threshold


def matched_sources(
    ocr_output: str, texts: Iterable[PlaceText], *, threshold: float
) -> set[TextSource]:
    """이 OCR 결과로 읽힌 문자열들의 출처 집합."""
    haystack = normalize_text(ocr_output)
    found: set[TextSource] = set()
    for entry in texts:
        if entry.source in found:
            continue
        if best_similarity(haystack, entry.normalized) >= threshold:
            found.add(entry.source)
    return found


# --------------------------------------------------------------------------- 정답 로드


def _parse_dark(raw: object, place_id: str) -> bool | None:
    if raw is None or raw == "auto":
        return None
    if isinstance(raw, bool):
        return raw
    raise TruthError(f'{place_id}: dark 는 true / false / "auto" 중 하나여야 합니다 ({raw!r})')


def _parse_texts(entry: dict[str, Any], place_id: str) -> list[PlaceText]:
    if "texts" in entry:
        raw_texts = entry["texts"]
        if not isinstance(raw_texts, list):
            raise TruthError(f"{place_id}: texts 는 리스트여야 합니다")
    elif "sign_text" in entry:
        raw_texts = [{"text": entry["sign_text"], "source": "sign"}]
    else:
        raise TruthError(f"{place_id}: texts(또는 v1 sign_text)가 없습니다")

    texts: list[PlaceText] = []
    for raw in raw_texts:
        if isinstance(raw, str):
            raw = {"text": raw, "source": "sign"}
        source = raw.get("source", "sign")
        if source not in TEXT_SOURCES:
            raise TruthError(f"{place_id}: source 는 {TEXT_SOURCES} 중 하나 ({source!r})")
        texts.append(PlaceText(text=str(raw.get("text", "")), source=source))
    return texts


def parse_truth(
    payload: dict[str, Any],
    *,
    clip: tuple[float, float] | None = None,
) -> tuple[list[PlaceTruth], list[str]]:
    """정답 JSON 을 읽어 (장소 목록, 경고 목록)을 돌려준다.

    Args:
        payload: JSON 객체.
        clip: 평가 영상이 원본의 `(start, end)` 구간을 잘라낸 클립이면 그 구간.
            구간을 클립 기준으로 옮기고, 클립 밖 구간은 버린다.

    Raises:
        TruthError: 형식 오류, 또는 문자열이 비어 있는 장소(라벨 미완)가 있을 때.
    """
    raw_places = payload.get("places")
    if not isinstance(raw_places, list) or not raw_places:
        raise TruthError("places 가 비어 있습니다")

    offset, clip_end = clip if clip is not None else (0.0, float("inf"))
    places: list[PlaceTruth] = []
    warnings: list[str] = []
    unlabeled: list[str] = []
    seen: set[str] = set()

    for index, entry in enumerate(raw_places):
        if not isinstance(entry, dict):
            raise TruthError(f"places[{index}] 가 객체가 아닙니다")
        place_id = str(entry.get("place_id") or f"p{index + 1}")
        if place_id in seen:
            raise TruthError(f"place_id 중복: {place_id}")
        seen.add(place_id)

        texts = [text for text in _parse_texts(entry, place_id) if text.normalized]
        name = str(entry.get("name") or "")
        if not texts:
            unlabeled.append(f"{place_id}({name})" if name else place_id)
            continue
        for text in texts:
            if len(text.normalized) < MIN_RELIABLE_LENGTH:
                warnings.append(f"{place_id}: '{text.text}' 는 너무 짧아 우연 일치가 쉽습니다")

        segments: list[tuple[float, float]] = []
        for raw_segment in entry.get("segments", []):
            if not isinstance(raw_segment, Sequence) or len(raw_segment) != 2:
                raise TruthError(f"{place_id}: segment 는 [시작, 끝] 이어야 합니다")
            start, end = parse_time(raw_segment[0]), parse_time(raw_segment[1])
            if end < start:
                raise TruthError(f"{place_id}: 끝이 시작보다 앞섭니다 {raw_segment!r}")
            start, end = max(start, offset), min(end, clip_end)
            if end <= start:
                continue
            segments.append((start - offset, end - offset))
        if not segments:
            warnings.append(f"{place_id}: 평가 구간 안에 segment 가 없어 제외합니다")
            continue

        places.append(
            PlaceTruth(
                place_id=place_id,
                name=name,
                texts=tuple(texts),
                dark=_parse_dark(entry.get("dark", "auto"), place_id),
                segments=tuple(segments),
            )
        )

    if unlabeled:
        raise TruthError(
            "화면 문자열(texts)이 비어 있는 장소가 있습니다 — 라벨을 채우거나 항목을 지우세요: "
            + ", ".join(unlabeled)
        )
    if not places:
        raise TruthError("평가 구간 안에 남는 장소가 없습니다")
    return places, warnings


def load_truth(
    path: Path, *, clip: tuple[float, float] | None = None
) -> tuple[list[PlaceTruth], list[str]]:
    return parse_truth(json.loads(path.read_text(encoding="utf-8")), clip=clip)


# --------------------------------------------------------------------------- 매니페스트

DEFAULT_FORMAT = "bv*[height<=720][vcodec^=avc1]/bv*[height<=720]"
DEFAULT_OCR_LANG = "kor+eng"
DEFAULT_JUDGE = "easyocr"


@dataclass(frozen=True)
class VideoSpec:
    """매니페스트의 영상 하나."""

    key: str
    url: str
    role: str
    clip: tuple[float, float] | None
    truth_path: Path
    format: str
    ocr_lang: str
    judge: str
    sha256: str | None
    split: str = "dev"

    @property
    def youtube_id(self) -> str | None:
        return extract_youtube_id(self.url)

    def video_path(self, cache_root: Path) -> Path:
        """평가에 쓰는 파일 — 클립이 없으면 원본, 있으면 잘라낸 클립."""
        folder = cache_root / self.key
        if self.clip is None:
            return folder / "source.mp4"
        start, end = self.clip
        return folder / f"clip_{int(start)}-{int(end)}.mp4"


_YOUTUBE_ID = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/))([A-Za-z0-9_-]{11})"
)
_KEY = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def extract_youtube_id(url: str) -> str | None:
    match = _YOUTUBE_ID.search(url)
    return match.group(1) if match else None


SPLITS = ("dev", "test")


def _parse_split(raw: object, key: str) -> str:
    """개발(dev) = 방법을 고르는 데 쓴 영상, 테스트(test) = 사전 등록 후 한 번만 평가하는 영상."""
    if raw not in SPLITS:
        raise TruthError(f"{key}: split 은 {SPLITS} 중 하나 ({raw!r})")
    return str(raw)


def select_specs(specs: list[VideoSpec], *, only: str = "", split: str = "") -> list[VideoSpec]:
    """--only(쉼표 구분 key) / --split 필터."""
    keys = {key.strip() for key in only.split(",") if key.strip()}
    chosen = [spec for spec in specs if (not keys or spec.key in keys)]
    if split:
        if split not in SPLITS:
            raise TruthError(f"--split 은 {SPLITS} 중 하나")
        chosen = [spec for spec in chosen if spec.split == split]
    if not chosen:
        raise TruthError(f"조건에 맞는 영상이 없습니다 (only={sorted(keys)}, split={split!r})")
    return chosen


def parse_manifest(payload: dict[str, Any], *, base_dir: Path) -> list[VideoSpec]:
    defaults = payload.get("defaults", {})
    raw_videos = payload.get("videos")
    if not isinstance(raw_videos, list) or not raw_videos:
        raise TruthError("manifest 에 videos 가 비어 있습니다")

    specs: list[VideoSpec] = []
    keys: set[str] = set()
    for index, raw in enumerate(raw_videos):
        key = str(raw.get("key", ""))
        if not _KEY.match(key):
            raise TruthError(f"videos[{index}].key 는 소문자·숫자·하이픈만: {key!r}")
        if key in keys:
            raise TruthError(f"key 중복: {key}")
        keys.add(key)

        url = str(raw.get("url", ""))
        if extract_youtube_id(url) is None:
            raise TruthError(f"{key}: 유튜브 URL 이 아닙니다 ({url!r})")

        clip_raw = raw.get("clip")
        clip: tuple[float, float] | None = None
        if clip_raw is not None:
            if not isinstance(clip_raw, list) or len(clip_raw) != 2:
                raise TruthError(f"{key}: clip 은 [시작, 끝] 또는 null")
            clip = (parse_time(clip_raw[0]), parse_time(clip_raw[1]))
            if clip[1] <= clip[0]:
                raise TruthError(f"{key}: clip 끝이 시작보다 앞섭니다")

        specs.append(
            VideoSpec(
                key=key,
                url=url,
                role=str(raw.get("role", "")),
                clip=clip,
                truth_path=base_dir / str(raw.get("truth", f"truth/{key}.json")),
                format=str(raw.get("format", defaults.get("format", DEFAULT_FORMAT))),
                ocr_lang=str(raw.get("ocr_lang", defaults.get("ocr_lang", DEFAULT_OCR_LANG))),
                judge=str(raw.get("judge", defaults.get("judge", DEFAULT_JUDGE))),
                split=_parse_split(raw.get("split", "dev"), key),
                sha256=raw.get("sha256"),
            )
        )
    return specs


def load_manifest(path: Path) -> list[VideoSpec]:
    return parse_manifest(json.loads(path.read_text(encoding="utf-8")), base_dir=path.parent)


def truth_template(
    chapters: Sequence[dict[str, Any]] | None, duration_sec: float
) -> dict[str, Any]:
    """유튜브 챕터로 라벨 초안을 만든다.

    챕터 제목은 **제작자가 붙인 이름**이라 판정에 쓰지 않는다(`name` 에만 넣는다).
    화면에 실제로 보이는 문자열(`texts`)은 사람이 영상을 보고 채워야 한다 — 챕터 제목
    "을지로 노포 맛집" 이 간판에 그렇게 적혀 있을 리 없다.
    """
    places: list[dict[str, Any]] = []
    for index, chapter in enumerate(chapters or []):
        start = float(chapter.get("start_time", 0.0))
        end = float(chapter.get("end_time", duration_sec))
        places.append(
            {
                "place_id": f"c{index + 1:02d}",
                "name": str(chapter.get("title", "")),
                "texts": [{"text": "", "source": "sign"}],
                "dark": "auto",
                "segments": [[format_time(start), format_time(end)]],
            }
        )
    if not places:
        places.append(
            {
                "place_id": "p1",
                "name": "",
                "texts": [{"text": "", "source": "sign"}],
                "dark": "auto",
                "segments": [["0:00", "0:10"]],
            }
        )
    return {
        "_how_to": (
            "texts 에는 화면에 실제로 보이는 문자열만 적는다(간판=sign, 편집 자막=caption). "
            "장소가 아닌 챕터(인트로·이동)는 항목을 지운다. segments 는 원본 영상 시각."
        ),
        "places": places,
    }
