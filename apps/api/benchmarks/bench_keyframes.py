"""비전 프론트엔드 벤치마크.

영상 하나를 받아 `select_keyframes`를 돌리고, 단계별 프레임 감축과 화질 지표 변화를
표로 출력한다. README에 붙일 컨택트시트와 보정 전/후 비교 이미지도 같이 만든다.

사용:

    python benchmarks/bench_keyframes.py path/to/video.mp4 --out-dir bench_out
    python benchmarks/bench_keyframes.py video.mp4 --embedder dinov3   # extras 설치 시

출력:
    bench_out/report.md            단계별 표 (README에 그대로 붙여넣기)
    bench_out/frames.csv           프레임별 원시 지표
    bench_out/contact_sheet.jpg    선별된 키프레임 격자
    bench_out/enhance_pairs.jpg    보정 전/후 비교 (가장 어두운 프레임 기준)
    bench_out/keyframes/*.jpg      실제 VLM 입력 프레임
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# 레포 루트에서 실행하지 않아도 app 패키지를 찾도록.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.vision import (  # noqa: E402
    KeyframeSelection,
    VisionFrontendConfig,
    build_embedder,
    compute_metrics,
    enhance_for_vlm,
    select_keyframes,
    textness,
)
from app.pipeline.vision.decode import read_bgr  # noqa: E402

# Gemini 이미지 입력 토큰 단가(가정값). 모델·해상도에 따라 달라지므로 CLI로 덮어쓸 수 있다.
DEFAULT_TOKENS_PER_IMAGE = 258


def _contact_sheet(paths: list[Path], out_path: Path, *, columns: int = 4, cell: int = 320) -> None:
    """선별된 키프레임을 격자 한 장으로 합친다."""
    if not paths:
        return
    rows = (len(paths) + columns - 1) // columns
    sheet = np.full((rows * cell, columns * cell, 3), 24, dtype=np.uint8)
    for order, path in enumerate(paths):
        image = read_bgr(path)
        height, width = image.shape[:2]
        scale = cell / max(height, width)
        resized = cv2.resize(image, (int(width * scale), int(height * scale)))
        row, column = divmod(order, columns)
        y_off = row * cell + (cell - resized.shape[0]) // 2
        x_off = column * cell + (cell - resized.shape[1]) // 2
        sheet[y_off : y_off + resized.shape[0], x_off : x_off + resized.shape[1]] = resized
        cv2.putText(
            sheet,
            path.stem.split("_t")[-1] + "s",
            (column * cell + 8, row * cell + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def _enhance_pairs(
    selection: KeyframeSelection,
    video_frames_dir: Path,
    out_path: Path,
    *,
    count: int = 3,
) -> list[dict[str, float]]:
    """보정 효과가 가장 큰 프레임들의 전/후를 나란히 붙이고 지표 변화를 반환한다."""
    darkest = sorted(selection.keyframes, key=lambda item: item.metrics.luma_mean)[:count]
    lapvars = [item.metrics.sharpness_lapvar for item in selection.frame_stats]
    sharpness_ref = float(np.percentile(lapvars, 90))

    panels: list[np.ndarray] = []
    deltas: list[dict[str, float]] = []
    for keyframe in darkest:
        source = next(
            (
                path
                for path in sorted(video_frames_dir.glob("frame_*.png"))
                if int(path.stem.split("_")[1]) == keyframe.frame_index + 1
            ),
            None,
        )
        if source is None:
            continue
        before = read_bgr(source)
        after, report = enhance_for_vlm(
            before,
            noise_sigma=keyframe.metrics.noise_sigma,
            sharpness_lapvar=keyframe.metrics.sharpness_lapvar,
            sharpness_ref=sharpness_ref,
        )
        before_metrics = compute_metrics(before, textness=textness(before).score)
        after_metrics = compute_metrics(after, textness=textness(after).score)
        deltas.append(
            {
                "timestamp_sec": keyframe.timestamp_sec,
                "luma_before": before_metrics.luma_mean,
                "luma_after": after_metrics.luma_mean,
                "rms_contrast_before": before_metrics.rms_contrast,
                "rms_contrast_after": after_metrics.rms_contrast,
                "lapvar_before": before_metrics.sharpness_lapvar,
                "lapvar_after": after_metrics.sharpness_lapvar,
                "textness_before": before_metrics.textness,
                "textness_after": after_metrics.textness,
                "stages": len(report.stages),
            }
        )

        target_h = 360
        scale = target_h / before.shape[0]
        size = (int(before.shape[1] * scale), target_h)
        pair = np.hstack([cv2.resize(before, size), cv2.resize(after, size)])
        cv2.putText(
            pair, "BEFORE", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            pair,
            "AFTER",
            (size[0] + 10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(pair)

    if panels:
        cv2.imwrite(str(out_path), np.vstack(panels), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return deltas


def _write_csv(selection: KeyframeSelection, out_path: Path) -> None:
    selected = {keyframe.frame_index for keyframe in selection.keyframes}
    reasons = {item.frame_index: item.reason.value for item in selection.rejected}
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "timestamp_sec",
                "lapvar",
                "tenengrad",
                "luma_mean",
                "clip_low",
                "clip_high",
                "rms_contrast",
                "noise_sigma",
                "colorfulness",
                "textness",
                "selected",
                "reject_reason",
            ]
        )
        for item in selection.frame_stats:
            metrics = item.metrics
            writer.writerow(
                [
                    item.index,
                    f"{item.timestamp_sec:.2f}",
                    f"{metrics.sharpness_lapvar:.2f}",
                    f"{metrics.sharpness_tenengrad:.1f}",
                    f"{metrics.luma_mean:.4f}",
                    f"{metrics.clipped_low_ratio:.4f}",
                    f"{metrics.clipped_high_ratio:.4f}",
                    f"{metrics.rms_contrast:.4f}",
                    f"{metrics.noise_sigma:.3f}",
                    f"{metrics.colorfulness:.2f}",
                    f"{metrics.textness:.3f}",
                    int(item.index in selected),
                    reasons.get(item.index, ""),
                ]
            )


def _report(
    selection: KeyframeSelection,
    deltas: list[dict[str, float]],
    *,
    video_path: Path,
    tokens_per_image: int,
) -> str:
    summary = selection.summary
    counts = Counter(item.reason.value for item in selection.rejected)
    decoded = summary.decoded_frames
    selected = summary.selected_frames

    def pct(value: int) -> str:
        return f"{value / decoded * 100:.1f}%" if decoded else "-"

    lines = [
        "## 비전 프론트엔드 벤치마크",
        "",
        f"- 입력: `{video_path.name}` ({summary.video_duration_sec:.1f}s)",
        f"- 샘플링: {selection.config.sample_fps} fps, 긴 변 {selection.config.long_edge_px}px",
        f"- 임베더: `{summary.embedder}`",
        f"- 처리 시간: {summary.elapsed_sec:.2f}s "
        f"({summary.elapsed_sec / max(summary.video_duration_sec, 1e-6):.3f}x 실시간)",
        "",
        "### 단계별 프레임 감축",
        "",
        "| 단계 | 프레임 | 원본 대비 | 비고 |",
        "|---|---:|---:|---|",
        f"| 디코딩 (샘플링) | {decoded} | 100.0% | {selection.config.sample_fps} fps |",
        f"| 저조도 구제 보정 | {summary.rescued_frames} | {pct(summary.rescued_frames)} | "
        f"WB+감마+CLAHE 후 게이트 재판정 |",
        f"| 화질 게이트 통과 | {summary.after_quality_gate} | {pct(summary.after_quality_gate)} | "
        f"블러 {counts.get('blur', 0)} · 언더 {counts.get('underexposed', 0)} · "
        f"오버 {counts.get('overexposed', 0)} 탈락 |",
        f"| 샷 분할 | {summary.shots_detected} | {pct(summary.shots_detected)} | "
        f"유사도 임계 {selection.config.scene_similarity_threshold} |",
        f"| 중복 샷 제거 후 | {summary.shots_detected - counts.get('duplicate', 0)} | "
        f"{pct(summary.shots_detected - counts.get('duplicate', 0))} | "
        f"재방문 컷 {counts.get('duplicate', 0)}개 병합 "
        f"(임계 {selection.config.duplicate_shot_threshold}) |",
        f"| **최종 VLM 입력** | **{selected}** | **{pct(selected)}** | "
        f"예산 컷 {counts.get('budget', 0)} |",
        "",
        f"**프레임 감축률 {summary.reduction_ratio * 100:.1f}%** ({decoded} → {selected}장)",
        "",
        "### VLM 입력 비용 (이미지 토큰 기준)",
        "",
        f"이미지 1장 = {tokens_per_image} 토큰 가정.",
        "",
        "| | 프레임 | 이미지 토큰 |",
        "|---|---:|---:|",
        f"| 전처리 없이 전부 투입 | {decoded} | {decoded * tokens_per_image:,} |",
        f"| 비전 프론트엔드 적용 | {selected} | {selected * tokens_per_image:,} |",
        f"| **절감** | **{decoded - selected}** | "
        f"**{(decoded - selected) * tokens_per_image:,}** |",
        "",
    ]

    if deltas:
        lines += [
            "### 미니 ISP 보정 효과 (가장 어두운 키프레임)",
            "",
            "| 시각 | 평균 휘도 | RMS 콘트라스트 | 선명도(lapvar) | 문자 점수 |",
            "|---|---|---|---|---|",
        ]
        for delta in deltas:
            lines.append(
                f"| {delta['timestamp_sec']:.1f}s "
                f"| {delta['luma_before']:.3f} → **{delta['luma_after']:.3f}** "
                f"| {delta['rms_contrast_before']:.3f} → **{delta['rms_contrast_after']:.3f}** "
                f"| {delta['lapvar_before']:.1f} → **{delta['lapvar_after']:.1f}** "
                f"| {delta['textness_before']:.3f} → **{delta['textness_after']:.3f}** |"
            )
        lines.append("")

    lines += [
        "### 선별된 키프레임",
        "",
        "| # | 시각 | 샷 | 점수 | 선명도 | 휘도 | 문자 점수 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for order, keyframe in enumerate(selection.keyframes, 1):
        lines.append(
            f"| {order} | {keyframe.timestamp_sec:.1f}s | {keyframe.shot_id} "
            f"| {keyframe.selection_score:.3f} | {keyframe.metrics.sharpness_lapvar:.1f} "
            f"| {keyframe.metrics.luma_mean:.3f} | {keyframe.metrics.textness:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="비전 프론트엔드 벤치마크")
    parser.add_argument("video", type=Path, help="입력 영상")
    parser.add_argument("--out-dir", type=Path, default=Path("bench_out"))
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-keyframes", type=int, default=16)
    parser.add_argument(
        "--embedder", choices=["perceptual", "dinov3", "auto"], default="perceptual"
    )
    parser.add_argument("--tokens-per-image", type=int, default=DEFAULT_TOKENS_PER_IMAGE)
    parser.add_argument("--no-enhance", action="store_true")
    parser.add_argument("--keep-frames", action="store_true", help="중간 프레임 PNG 보존")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"영상을 찾을 수 없습니다: {args.video}", file=sys.stderr)
        return 1

    out_dir: Path = args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    config = VisionFrontendConfig(
        sample_fps=args.sample_fps,
        max_keyframes=args.max_keyframes,
        enhance=not args.no_enhance,
    )
    selection = select_keyframes(
        args.video, out_dir, config=config, embedder=build_embedder(args.embedder)
    )

    _write_csv(selection, out_dir / "frames.csv")
    _contact_sheet(
        [keyframe.path for keyframe in selection.keyframes], out_dir / "contact_sheet.jpg"
    )
    deltas = (
        _enhance_pairs(selection, out_dir / "frames", out_dir / "enhance_pairs.jpg")
        if config.enhance
        else []
    )
    report = _report(
        selection,
        deltas,
        video_path=args.video,
        tokens_per_image=args.tokens_per_image,
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        selection.summary.model_dump_json(indent=2), encoding="utf-8"
    )

    if not args.keep_frames:
        shutil.rmtree(out_dir / "frames", ignore_errors=True)

    print(report)
    print(f"\n→ 결과: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
