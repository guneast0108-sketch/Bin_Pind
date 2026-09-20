"""장소 보존률 정답 형식 · 문자열 판정 · 유튜브 매니페스트 (benchmarks/recall_truth.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from recall_truth import (  # noqa: E402
    PlaceText,
    TruthError,
    extract_youtube_id,
    format_time,
    matched_sources,
    normalize_text,
    parse_manifest,
    parse_time,
    parse_truth,
    select_specs,
    text_is_readable,
    truth_template,
)

# --------------------------------------------------------------------------- 시각


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(12, 12.0), (12.5, 12.5), ("12.5", 12.5), ("5:12", 312.0), ("1:05:12", 3912.0)],
)
def test_parse_time_accepts_seconds_and_player_format(raw: object, expected: float) -> None:
    assert parse_time(raw) == expected


@pytest.mark.parametrize("raw", ["", "5:", "a:10", "1:2:3:4", True, None, -1])
def test_parse_time_rejects_garbage(raw: object) -> None:
    with pytest.raises(TruthError):
        parse_time(raw)


def test_format_time_matches_youtube_player() -> None:
    assert format_time(312.9) == "5:12"
    assert format_time(3912) == "1:05:12"


# --------------------------------------------------------------------------- 문자열


def test_normalize_keeps_hangul() -> None:
    """예전 정규화([^A-Z0-9] 제거)는 한글을 전부 지워 빈 문자열이 됐다."""
    assert normalize_text("블루보틀 성수") != ""
    assert normalize_text("Cafe Lumière!") == "CAFELUMIERE"


def test_hangul_compared_at_jamo_level() -> None:
    # 받침 하나 틀림(식당→식단): 자모 12개 중 1개 차이 → 통과
    assert text_is_readable("용강식단 영업중", "용강식당", threshold=0.80)
    # 음절 둘이 통째로 틀림 → 탈락
    assert not text_is_readable("용강분식", "용강식당", threshold=0.80)


def test_compatibility_jamo_from_ocr_is_folded() -> None:
    """OCR이 가끔 내놓는 호환 자모(ㅇ U+3147)도 조합형 자모와 같게 본다."""
    assert normalize_text("ㅇ") == normalize_text("ᄋ")


def test_english_behaviour_unchanged() -> None:
    assert text_is_readable("care Lumiere 1998", "CAFE LUMIERE", threshold=0.80)
    assert not text_is_readable("BAKERY", "CAFE LUMIERE", threshold=0.80)


def test_matched_sources_reports_where_text_was_read() -> None:
    texts = (PlaceText("GRANDMA CABINET", "sign"), PlaceText("그랜마캐비넷", "caption"))
    assert matched_sources("오늘은 그랜마캐비넷 방문", texts, threshold=0.8) == {"caption"}
    assert matched_sources("GRANDMA CABINET", texts, threshold=0.8) == {"sign"}
    assert matched_sources("nothing", texts, threshold=0.8) == set()


# --------------------------------------------------------------------------- 정답


def test_v1_truth_still_loads() -> None:
    places, warnings = parse_truth(
        {"places": [{"place_id": "p1", "sign_text": "CAFE", "dark": True, "segments": [[0, 5]]}]}
    )
    assert places[0].texts == (PlaceText("CAFE", "sign"),)
    assert places[0].dark is True
    assert places[0].has_sign
    assert warnings == []


def test_v2_truth_with_clip_offset() -> None:
    payload = {
        "places": [
            {
                "place_id": "a",
                "name": "용강식당",
                "texts": [{"text": "용강식당", "source": "caption"}],
                "segments": [["1:50", "2:10"], ["9:00", "9:30"]],
            },
            {"place_id": "outside", "texts": ["OUTSIDE"], "segments": [["20:00", "20:10"]]},
        ]
    }
    places, warnings = parse_truth(payload, clip=(120.0, 600.0))
    assert [place.place_id for place in places] == ["a"]
    # 1:50–2:10 은 클립 시작(2:00)에서 잘려 0–10초, 9:00–9:30 은 클립 끝(10:00) 안 → 420–450
    assert places[0].segments == ((0.0, 10.0), (420.0, 450.0))
    assert places[0].dark is None  # 기본 auto
    assert not places[0].has_sign
    assert any("outside" in warning for warning in warnings)


def test_unlabeled_places_fail_loudly() -> None:
    """초안 그대로 돌리면 모든 장소가 조용히 '읽기 실패'가 된다 — 그 전에 멈춘다."""
    with pytest.raises(TruthError, match="c01"):
        parse_truth(truth_template([{"title": "용강식당", "start_time": 0, "end_time": 60}], 60))


def test_short_text_warns() -> None:
    _, warnings = parse_truth({"places": [{"texts": ["CU"], "segments": [[0, 1]]}]})
    assert any("너무 짧아" in warning for warning in warnings)


@pytest.mark.parametrize(
    "entry",
    [
        {"texts": ["A CAFE"], "segments": [[5, 1]]},
        {"texts": [{"text": "A CAFE", "source": "banner"}], "segments": [[0, 1]]},
        {"texts": ["A CAFE"], "dark": "maybe", "segments": [[0, 1]]},
    ],
)
def test_invalid_truth_rejected(entry: dict[str, object]) -> None:
    with pytest.raises(TruthError):
        parse_truth({"places": [entry]})


def test_template_uses_chapters_for_segments_only() -> None:
    template = truth_template(
        [{"title": "인트로", "start_time": 0.0, "end_time": 42.0}], duration_sec=600.0
    )
    place = template["places"][0]
    assert place["name"] == "인트로"
    assert place["texts"] == [{"text": "", "source": "sign"}]  # 챕터 제목을 정답으로 쓰지 않는다
    assert place["segments"] == [["0:00", "0:42"]]


# --------------------------------------------------------------------------- 매니페스트


@pytest.mark.parametrize(
    ("url", "video_id"),
    [
        ("https://www.youtube.com/watch?v=RkhW3kAMP2E", "RkhW3kAMP2E"),
        ("https://youtube.com/watch?feature=share&v=RkhW3kAMP2E", "RkhW3kAMP2E"),
        ("https://youtu.be/RkhW3kAMP2E?t=30", "RkhW3kAMP2E"),
        ("https://www.youtube.com/shorts/_vu0c2nzKjE", "_vu0c2nzKjE"),
        ("https://vimeo.com/123", None),
    ],
)
def test_extract_youtube_id(url: str, video_id: str | None) -> None:
    assert extract_youtube_id(url) == video_id


def test_manifest_defaults_and_paths(tmp_path: Path) -> None:
    specs = parse_manifest(
        {
            "defaults": {"ocr_lang": "kor"},
            "videos": [
                {"key": "night", "url": "https://youtu.be/yArKEcqEYgk", "clip": ["1:00", "6:00"]},
                {"key": "day", "url": "https://youtu.be/CqjUMq6BbDA", "ocr_lang": "eng"},
            ],
        },
        base_dir=tmp_path,
    )
    night, day = specs
    assert night.clip == (60.0, 360.0)
    assert night.ocr_lang == "kor"
    assert night.truth_path == tmp_path / "truth" / "night.json"
    assert (
        night.video_path(tmp_path / ".cache") == tmp_path / ".cache" / "night" / "clip_60-360.mp4"
    )
    assert night.judge == "easyocr"  # 유튜브 기본 판정자
    assert day.ocr_lang == "eng"
    assert day.video_path(tmp_path / ".cache").name == "source.mp4"


@pytest.mark.parametrize(
    "videos",
    [
        [],
        [{"key": "Bad Key", "url": "https://youtu.be/yArKEcqEYgk"}],
        [{"key": "a", "url": "https://example.com/v"}],
        [{"key": "a", "url": "https://youtu.be/yArKEcqEYgk", "clip": ["6:00", "1:00"]}],
        [
            {"key": "a", "url": "https://youtu.be/yArKEcqEYgk"},
            {"key": "a", "url": "https://youtu.be/CqjUMq6BbDA"},
        ],
    ],
)
def test_manifest_rejects_invalid(videos: list[dict[str, object]], tmp_path: Path) -> None:
    with pytest.raises(TruthError):
        parse_manifest({"videos": videos}, base_dir=tmp_path)


def test_committed_manifest_is_valid() -> None:
    manifest = Path(__file__).resolve().parents[2] / "benchmarks" / "youtube" / "manifest.json"

    specs = parse_manifest(
        json.loads(manifest.read_text(encoding="utf-8")), base_dir=manifest.parent
    )
    assert len(specs) >= 2
    assert all(spec.youtube_id for spec in specs)


def test_split_filter(tmp_path: Path) -> None:
    specs = parse_manifest(
        {
            "videos": [
                {"key": "a", "url": "https://youtu.be/yArKEcqEYgk"},
                {"key": "b", "url": "https://youtu.be/CqjUMq6BbDA", "split": "test"},
            ]
        },
        base_dir=tmp_path,
    )
    assert specs[0].split == "dev"  # 기본값
    assert [s.key for s in select_specs(specs, split="test")] == ["b"]
    assert [s.key for s in select_specs(specs, only="a,b")] == ["a", "b"]
    with pytest.raises(TruthError):
        select_specs(specs, only="a", split="test")
    with pytest.raises(TruthError):
        parse_manifest(
            {"videos": [{"key": "c", "url": "https://youtu.be/CqjUMq6BbDA", "split": "val"}]},
            base_dir=tmp_path,
        )


def test_committed_manifest_has_test_split() -> None:
    manifest = Path(__file__).resolve().parents[2] / "benchmarks" / "youtube" / "manifest.json"
    specs = parse_manifest(
        json.loads(manifest.read_text(encoding="utf-8")), base_dir=manifest.parent
    )
    dev = {spec.youtube_id for spec in specs if spec.split == "dev"}
    test = {spec.youtube_id for spec in specs if spec.split == "test"}
    assert len(dev) == 3 and len(test) == 3
    assert not dev & test
