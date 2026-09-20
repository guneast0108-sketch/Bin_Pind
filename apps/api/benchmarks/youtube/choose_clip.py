"""사전 등록 보완 규칙으로 테스트 영상의 평가 구간을 고른다 (PREREGISTRATION.md 「평가 구간」).

    python benchmarks/youtube/choose_clip.py benchmarks/youtube/truth/<key>.json --duration 1116

영상 전체를 라벨한 정답 파일을 받아, 길이 880초 창을 1초 단위로 밀며 segment 가 창과 2초 이상 겹치는
장소 수가 최대인 창을 고른다(동점이면 이른 창). 판단이 끼지 않게 규칙을 코드로 고정해 둔다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recall_truth import format_time, parse_time  # noqa: E402

WINDOW_SEC = 880
MIN_OVERLAP_SEC = 2.0
MIN_PLACES = 4


def choose(places: list[dict[str, object]], duration: float) -> tuple[int, int, list[str]]:
    spans = [
        (str(place["place_id"]), [(parse_time(a), parse_time(b)) for a, b in place["segments"]])  # type: ignore[union-attr,misc]
        for place in places
    ]
    if duration <= WINDOW_SEC:
        return 0, int(duration), [pid for pid, _ in spans]
    best: tuple[int, list[str]] = (0, [])
    for start in range(0, int(duration - WINDOW_SEC) + 1):
        end = start + WINDOW_SEC
        inside = [
            pid
            for pid, segments in spans
            if any(min(b, end) - max(a, start) >= MIN_OVERLAP_SEC for a, b in segments)
        ]
        if len(inside) > len(best[1]):
            best = (start, inside)
    return best[0], best[0] + WINDOW_SEC, best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("truth", type=Path)
    parser.add_argument("--duration", type=float, required=True)
    args = parser.parse_args()
    payload = json.loads(args.truth.read_text(encoding="utf-8"))
    start, end, inside = choose(payload["places"], args.duration)
    verdict = "OK" if len(inside) >= MIN_PLACES else "제외 기준 해당 (장소 4곳 미만)"
    print(
        json.dumps(
            {
                "clip": [format_time(start), format_time(end)],
                "clip_sec": [start, end],
                "places_in_clip": inside,
                "count": len(inside),
                "verdict": verdict,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
