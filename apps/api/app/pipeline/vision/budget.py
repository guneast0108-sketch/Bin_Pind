"""예산 컷 — "문자가 보이는 프레임을 시간적으로 흩어서" 고른다.

`dedup.select_diverse_budget`(k-center, 시각적 다양성)는 합성 장면에서 한 장소가 슬롯을
독식하는 문제를 고쳤지만, 유튜브 실영상에서는 점수를 시작점에만 쓰는 구조 탓에 **문자가
없는 프레임도 똑같이 뽑았다.** 실영상에서 장소 문자열이 읽히는 프레임은 전체의 4–7%뿐이라,
시각적으로 다양한 16장은 대부분 음식·사람·풍경이었다.

여기서는 프레임을 문자 검출 점수 순으로 보면서, 이미 고른 프레임과 시간상 가까운 것은
건너뛴다(시간 NMS). 간격은 "예산을 영상 전체에 고르게 폈을 때 간격의 절반"이다 — 한 장소에
머무는 동안의 연속 프레임이 슬롯을 여러 개 먹지 않게 하면서, 장소가 짧게 연달아 나오는
구간도 둘 다 잡을 수 있는 폭이다.

이 규칙(점수 = 검출 박스 수, 간격 비율 0.5, 동점은 앞 프레임)은 유튜브 개발 세트 3편에서
정했고(16장에 8/18, 무작위 2.7), **사전 등록한 테스트 3편에서는 재현되지 않아 기각됐다**
(35곳 중 4, 기본 diverse 3, 무작위 기댓값 3.9). 박스 수 상위 프레임은 메뉴판·진열대처럼 문자가
빽빽한 프레임이라 장소 이름 프레임과 달랐다. 실험 재현을 위해 남긴다
(`benchmarks/youtube/PREREGISTRATION.md`, `docs/vision-frontend-decisions.md`).
"""

from __future__ import annotations

from collections.abc import Sequence


def temporal_gap_frames(total_sampled: int, budget: int, gap_ratio: float) -> int:
    """고른 프레임끼리 떨어져야 하는 최소 간격(샘플 인덱스 단위, 이 값 **초과**)."""
    if budget <= 0 or total_sampled <= 0:
        return 0
    return int(gap_ratio * total_sampled / budget)


def select_text_nms(
    frame_indices: Sequence[int],
    scores: Sequence[float],
    *,
    budget: int,
    min_gap: int,
) -> list[int]:
    """점수 내림차순으로 보며 시간 간격이 `min_gap` 을 **넘는** 프레임만 남긴다.

    Args:
        frame_indices: 후보 프레임의 샘플 인덱스(오름차순일 필요 없음).
        scores: 후보별 점수. 클수록 먼저 고른다. 동점이면 인덱스가 작은 쪽.
        budget: 최대 선택 수. 간격 제약 때문에 이보다 적게 고를 수 있다.
        min_gap: 이미 고른 프레임과의 인덱스 차가 이 값 이하이면 건너뛴다.

    Returns:
        고른 프레임 인덱스(오름차순).
    """
    if len(frame_indices) != len(scores):
        raise ValueError("frame_indices 와 scores 길이가 다릅니다")
    if budget <= 0:
        return []
    order = sorted(range(len(frame_indices)), key=lambda i: (-scores[i], frame_indices[i]))
    chosen: list[int] = []
    for position in order:
        candidate = frame_indices[position]
        if all(abs(candidate - kept) > min_gap for kept in chosen):
            chosen.append(candidate)
            if len(chosen) == budget:
                break
    return sorted(chosen)
