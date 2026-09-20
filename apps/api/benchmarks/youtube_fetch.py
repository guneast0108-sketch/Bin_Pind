"""유튜브 검증 영상 준비 — 다운로드 · 클립 · 라벨링용 콘택트 시트 · 정답 초안.

## 왜 직접 찍은 영상 대신 유튜브인가

Pind 가 실제로 받는 입력은 유튜브 여행 영상이다. 직접 찍은 영상은 원본 화질(고비트레이트,
편집 없음)이라 **서비스 입력과 분포가 다르다.** 유튜브 영상에는 합성 장면에도, 직접 찍은
영상에도 없는 조건이 있다.

- **재압축** — 720p H.264 기준 수 Mbps. 블록 아티팩트가 저조도 노이즈와 섞인다.
- **편집 컷** — 하드 컷·점프 컷이 많아 "인접 프레임 유사도" 분포가 촬영 원본과 다르다.
- **편집 자막** — 상호가 자막으로 들어간다. 파이프라인의 문자 saliency 가 간판 대신 자막을
  고를 수 있고, 자막은 화질 보정 없이도 읽힌다.
- **색보정·HDR 톤매핑** — 제작자가 이미 밝기·대비를 만졌다.

## 재현성

영상 파일은 저작권 때문에 레포에 넣지 않는다(`benchmarks/youtube/.cache/`, gitignore).
대신 **URL · 포맷 선택자 · 구간 · 정답**을 매니페스트로 커밋하고, 받은 파일의 포맷 id ·
해상도 · sha256 을 `info.json` 과 보고서에 남긴다. 매니페스트에 `sha256` 을 적어 두면
유튜브가 재인코딩했거나 영상이 바뀐 경우 경고한다.

## 사용

    # 1) 다운로드 + 콘택트 시트 + 정답 초안
    python benchmarks/youtube_fetch.py                  # 매니페스트의 모든 영상
    python benchmarks/youtube_fetch.py --only euljiro-night-walk

    # 2) 콘택트 시트(.cache/<key>/sheets/)를 보며 truth/<key>.json 의 texts 를 채운다

    # 3) 평가
    python benchmarks/validate_recall.py --manifest benchmarks/youtube/manifest.json

유튜브 접속이 필요하다. 최신 yt-dlp 는 유튜브 서명 해독에 JS 런타임을 요구할 수 있다 —
`Requested format is not available` · `n challenge` 오류가 나면 `pip install -U yt-dlp` 후
`brew install deno`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recall_truth import (  # noqa: E402
    TruthError,
    VideoSpec,
    format_time,
    load_manifest,
    select_specs,
    truth_template,
)

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "youtube" / "manifest.json"

# 콘택트 시트 — 16:9 썸네일 4x4 = 1920x1080 한 장. 3초 간격이면 시트 한 장이 48초.
THUMB_W, THUMB_H = 480, 270
GRID_COLS, GRID_ROWS = 4, 4

# 클립 재인코딩 품질. 유튜브 원본 위에 한 번 더 압축하는 셈이라 거의 무손실(CRF 12)로 둔다.
CLIP_CRF = "12"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(payload: object) -> str:
    """손으로 고치기 쉬운 JSON — 구간 쌍과 문자열 항목을 한 줄로 접는다."""
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    scalar = r'(?:"[^"\\]*"|-?\d+(?:\.\d+)?)'
    text = re.sub(rf"\[\s+({scalar}),\s+({scalar})\s+\]", r"[\1, \2]", text)
    return re.sub(
        rf'\{{\s+"text": ({scalar}),\s+"source": ({scalar})\s+\}}',
        r'{"text": \1, "source": \2}',
        text,
    )


# --------------------------------------------------------------------------- 다운로드


def download(spec: VideoSpec, folder: Path) -> dict[str, Any]:
    """영상 트랙만 받는다(오디오 불필요). 이미 있으면 메타데이터만 다시 읽는다."""
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise SystemExit("yt-dlp 가 필요합니다: pip install -U yt-dlp") from exc

    folder.mkdir(parents=True, exist_ok=True)
    existing = sorted(folder.glob("download.*"))
    options: dict[str, Any] = {
        "format": spec.format,
        "outtmpl": str(folder / "download.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "skip_download": bool(existing),
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(spec.url, download=not existing)
        version = yt_dlp.version.__version__

    files = sorted(folder.glob("download.*"))
    if not files:
        raise SystemExit(f"{spec.key}: 다운로드된 파일이 없습니다")
    downloaded = files[0]

    source = folder / "source.mp4"
    if not source.exists():
        if downloaded.suffix == ".mp4":
            downloaded.rename(source)
        else:
            # webm(VP9/AV1) 등은 컨테이너만 mp4 로 바꾼다. 재인코딩하지 않는다.
            _ffmpeg(["-i", str(downloaded), "-c", "copy", "-an", str(source)])
            downloaded.unlink()
    elif downloaded.exists() and downloaded != source:
        downloaded.unlink()

    return {
        "key": spec.key,
        "url": spec.url,
        "id": info.get("id"),
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "upload_date": info.get("upload_date"),
        "duration": info.get("duration"),
        "chapters": info.get("chapters") or [],
        "format_selector": spec.format,
        "format_id": info.get("format_id"),
        "width": info.get("width"),
        "height": info.get("height"),
        "fps": info.get("fps"),
        "vcodec": info.get("vcodec"),
        "tbr_kbps": info.get("tbr"),
        "yt_dlp": version,
    }


def _ffmpeg(arguments: list[str]) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if completed.returncode != 0:
        raise SystemExit(f"ffmpeg 실패: {completed.stderr[:500]}")


def cut_clip(source: Path, target: Path, start: float, end: float) -> None:
    """정확한 시각으로 자른다.

    `-c copy` 로 자르면 키프레임 경계로 밀려 정답 구간 시각이 수 초씩 어긋난다.
    그래서 재인코딩하되 거의 무손실로 한다.
    """
    if target.exists():
        return
    _ffmpeg(
        [
            "-ss", f"{start:.3f}",
            "-i", str(source),
            "-t", f"{end - start:.3f}",
            "-an",
            "-c:v", "libx264", "-crf", CLIP_CRF, "-preset", "fast", "-pix_fmt", "yuv420p",
            str(target),
        ]
    )  # fmt: skip


# --------------------------------------------------------------------------- 콘택트 시트


def contact_sheets(
    video: Path,
    out_dir: Path,
    *,
    interval_sec: float,
    time_offset: float,
    chapters: list[dict[str, Any]],
) -> int:
    """라벨링용 콘택트 시트. 썸네일마다 **원본 영상 시각**을 찍는다.

    정답 JSON 의 segments 도 원본 시각으로 적으므로, 시트에 보이는 숫자를 그대로 옮겨
    적으면 된다. 작은 간판은 썸네일에서 안 읽힐 수 있다 — 그 시각을 유튜브에서 직접 확인한다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sheet_*.jpg"):
        old.unlink()

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps * interval_sec)), 1)
    per_sheet = GRID_COLS * GRID_ROWS

    thumbs: list[tuple[float, np.ndarray]] = []
    sheet_count = 0
    index_lines = ["| 시트 | 원본 시각 | 이 구간의 챕터 |", "|---|---|---|"]

    def flush() -> None:
        nonlocal sheet_count
        if not thumbs:
            return
        canvas = np.zeros((THUMB_H * GRID_ROWS, THUMB_W * GRID_COLS, 3), dtype=np.uint8)
        for slot, (timestamp, thumb) in enumerate(thumbs):
            row, col = divmod(slot, GRID_COLS)
            y, x = row * THUMB_H, col * THUMB_W
            canvas[y : y + THUMB_H, x : x + THUMB_W] = thumb
            label = format_time(timestamp)
            cv2.rectangle(canvas, (x, y), (x + 92, y + 30), (0, 0, 0), -1)
            cv2.putText(
                canvas, label, (x + 6, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA,
            )  # fmt: skip
            cv2.rectangle(canvas, (x, y), (x + THUMB_W - 1, y + THUMB_H - 1), (40, 40, 40), 1)
        name = f"sheet_{sheet_count:03d}.jpg"
        cv2.imwrite(str(out_dir / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        first, last = thumbs[0][0], thumbs[-1][0]
        overlapping = [
            str(chapter.get("title", ""))
            for chapter in chapters
            if float(chapter.get("start_time", 0)) <= last
            and float(chapter.get("end_time", math.inf)) >= first
        ]
        index_lines.append(
            f"| {name} | {format_time(first)}–{format_time(last)} | "
            f"{' / '.join(overlapping) or '—'} |"
        )
        sheet_count += 1
        thumbs.clear()

    frame_index = 0
    while capture.grab():
        if frame_index % step == 0:
            ok, frame = capture.retrieve()
            if ok:
                thumb = cv2.resize(frame, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
                thumbs.append((frame_index / fps + time_offset, thumb))
                if len(thumbs) == per_sheet:
                    flush()
        frame_index += 1
    flush()
    capture.release()

    (out_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return sheet_count


# --------------------------------------------------------------------------- 진입점


def prepare(spec: VideoSpec, cache_root: Path, *, interval_sec: float, sheets: bool) -> None:
    folder = cache_root / spec.key
    print(f"\n=== {spec.key} · {spec.url}")
    meta = download(spec, folder)
    source = folder / "source.mp4"
    meta["sha256"] = sha256_of(source)
    meta["fetched_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["clip"] = list(spec.clip) if spec.clip else None
    (folder / "info.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"  {meta['title']} · {meta['channel']} · {format_time(float(meta['duration'] or 0))}"
        f" · 포맷 {meta['format_id']} {meta['width']}x{meta['height']} {meta['vcodec']}"
    )
    if spec.sha256 and spec.sha256 != meta["sha256"]:
        print(
            "  경고: sha256 이 매니페스트와 다릅니다 — 유튜브가 재인코딩했거나 포맷 선택이"
            " 달라졌습니다. 수치를 이전 결과와 직접 비교하지 마세요."
        )

    target = spec.video_path(cache_root)
    if spec.clip is not None:
        start, end = spec.clip
        duration = float(meta["duration"] or 0)
        if duration and end > duration:
            raise SystemExit(f"{spec.key}: clip 끝({end})이 영상 길이({duration})를 넘습니다")
        cut_clip(source, target, start, end)
        print(f"  클립 {format_time(start)}–{format_time(end)} → {target.name}")

    if not spec.truth_path.exists():
        spec.truth_path.parent.mkdir(parents=True, exist_ok=True)
        template = truth_template(meta["chapters"], float(meta["duration"] or 0))
        spec.truth_path.write_text(compact_json(template) + "\n", encoding="utf-8")
        print(
            f"  정답 초안 → {spec.truth_path} (챕터 {len(meta['chapters'])}개로 구간만 채움,"
            " texts 는 직접)"
        )

    if sheets:
        count = contact_sheets(
            target,
            folder / "sheets",
            interval_sec=interval_sec,
            time_offset=spec.clip[0] if spec.clip else 0.0,
            chapters=meta["chapters"],
        )
        print(f"  콘택트 시트 {count}장 → {folder / 'sheets'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="유튜브 검증 영상 준비")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--only", default="", help="이 key 들만 (쉼표 구분)")
    parser.add_argument("--split", default="", help="dev | test 만 (매니페스트)")
    parser.add_argument("--sheet-interval", type=float, default=3.0, help="썸네일 간격(초)")
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args()

    try:
        specs = load_manifest(args.manifest)
    except TruthError as exc:
        raise SystemExit(f"매니페스트 오류: {exc}") from exc
    try:
        specs = select_specs(specs, only=args.only, split=args.split)
    except TruthError as exc:
        raise SystemExit(str(exc)) from exc

    cache_root = args.manifest.parent / ".cache"
    for spec in specs:
        prepare(spec, cache_root, interval_sec=args.sheet_interval, sheets=not args.no_sheets)
    print("\n다음: 시트를 보며 truth/*.json 의 texts 를 채운 뒤")
    print(f"  python benchmarks/validate_recall.py --manifest {args.manifest}")


if __name__ == "__main__":
    main()
