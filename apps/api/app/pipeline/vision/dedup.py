"""장면 분할과 중복 제거.

브이로그는 한 가게에서 수십 초를 머문다. 1 fps로 뽑으면 거의 동일한 프레임이 수십 장 나오고,
이걸 그대로 VLM에 넣으면 토큰만 쓰고 새 정보는 얻지 못한다. 여기서는 임베딩 코사인 유사도로
연속 프레임을 샷으로 묶고(온라인 1-pass), 샷마다 대표 1장만 남긴다.

샷 분할은 인접 프레임 유사도가 임계 아래로 떨어지는 지점을 컷으로 보는 고전적 방식이다
(Yeung & Liu, "Efficient Matching and Clustering of Video Shots", ICIP 1995의 인접 유사도
기반 분할을 단순화). 핸드헬드 흔들림 때문에 유사도가 한 프레임만 튀는 경우가 많아,
`min_shot_gap_sec`로 최소 샷 길이를 두어 과분할을 막는다.
"""

from __future__ import annotations

import numpy as np
import structlog
from numpy.typing import NDArray

from app.pipeline.vision.types import ShotSegment

logger = structlog.get_logger(__name__)

Embedding = NDArray[np.float32]


def cosine_similarity_matrix(embeddings: Embedding) -> NDArray[np.float32]:
    """L2 정규화된 임베딩들의 전체 코사인 유사도 행렬."""
    return (embeddings @ embeddings.T).astype(np.float32)


def adjacent_similarities(embeddings: Embedding) -> NDArray[np.float32]:
    """인접 프레임 간 코사인 유사도. 길이는 N-1."""
    if len(embeddings) < 2:
        return np.zeros(0, dtype=np.float32)
    sims: NDArray[np.float32] = np.sum(embeddings[:-1] * embeddings[1:], axis=1).astype(np.float32)
    return sims


def adaptive_scene_threshold(
    embeddings: Embedding,
    *,
    default_threshold: float,
    max_cut_rate: float = 0.25,
) -> float:
    """영상 자신의 인접 유사도 분포에서 샷 경계 임계를 정한다.

    고정 임계는 영상 사이를 넘어가지 못한다. 삼각대에 올린 영상은 인접 유사도가
    0.99 근처에 몰리지만, 손에 들고 천천히 패닝하면 **같은 장소 안에서도** 인접
    유사도가 0.75까지 떨어진다. 그 영상에 0.88을 적용하면 한 장소가 두세 개의 샷으로
    쪼개지고, 예산이 그 쪼개진 샷들에 흩어져 다른 장소가 한 장도 못 남는다
    (실측: 장소 8곳이 19개 샷으로 분할, 장소당 프레임 2.7장, 잃은 장소 2곳).

    그래서 "몇 쌍을 경계로 볼지"를 정하고 임계를 거기서 역산한다. 인접 유사도의
    하위 `max_cut_rate` 분위를 임계로 쓰면 경계 비율이 그 값을 넘지 않는다. 기본
    임계보다 높아지는 경우는 쓰지 않으므로(min), 정적인 영상에서는 기존 동작이
    그대로 유지되고 패닝이 많은 영상에서만 느슨해진다.

    Args:
        embeddings: (N, D) 시간순 L2 정규화 임베딩.
        default_threshold: 임베더 권장 임계. 상한으로 쓴다.
        max_cut_rate: 경계로 삼을 인접쌍의 최대 비율.

    Returns:
        실제로 쓸 유사도 임계.
    """
    sims = adjacent_similarities(embeddings)
    if sims.size < 4:
        return default_threshold

    # method="lower"를 쓰는 이유: 보간된 분위값은 임계 바로 위의 값까지 컷으로 만들어
    # 경계 비율이 max_cut_rate를 넘길 수 있다. 하위 값을 그대로 쓰면 상한이 보장된다.
    quantile = float(np.percentile(sims.astype(np.float64), max_cut_rate * 100.0, method="lower"))
    threshold = min(default_threshold, quantile)
    if threshold < default_threshold:
        logger.info(
            "scene_threshold_adapted",
            default=round(default_threshold, 3),
            adapted=round(threshold, 3),
            mean_adjacent_similarity=round(float(sims.mean()), 3),
            max_cut_rate=max_cut_rate,
        )
    return threshold


def segment_shots(
    embeddings: Embedding,
    timestamps: list[float],
    frame_indices: list[int],
    *,
    similarity_threshold: float = 0.88,
    min_shot_gap_sec: float = 1.5,
) -> list[ShotSegment]:
    """연속 프레임을 샷으로 묶는다.

    Args:
        embeddings: (N, D) L2 정규화 임베딩.
        timestamps: 각 프레임의 영상 내 시각(초).
        frame_indices: 각 프레임의 원본 샘플 인덱스.
        similarity_threshold: 이 값 이상이면 같은 샷.
        min_shot_gap_sec: 새 샷을 열 수 있는 최소 경과 시간.

    Returns:
        시간순 `ShotSegment` 리스트.
    """
    count = len(frame_indices)
    if count == 0:
        return []
    if not (count == len(timestamps) == len(embeddings)):
        raise ValueError("embeddings, timestamps, frame_indices 길이가 일치해야 합니다")

    sims = adjacent_similarities(embeddings)
    shots: list[ShotSegment] = []
    current: list[int] = [0]
    shot_start_time = timestamps[0]

    for position in range(1, count):
        similar = bool(sims[position - 1] >= similarity_threshold)
        long_enough = (timestamps[position] - shot_start_time) >= min_shot_gap_sec
        if similar or not long_enough:
            current.append(position)
            continue

        shots.append(
            ShotSegment(
                shot_id=len(shots),
                start_sec=timestamps[current[0]],
                end_sec=timestamps[current[-1]],
                frame_indices=[frame_indices[pos] for pos in current],
            )
        )
        current = [position]
        shot_start_time = timestamps[position]

    shots.append(
        ShotSegment(
            shot_id=len(shots),
            start_sec=timestamps[current[0]],
            end_sec=timestamps[current[-1]],
            frame_indices=[frame_indices[pos] for pos in current],
        )
    )

    logger.info(
        "shots_segmented",
        frames=count,
        shots=len(shots),
        threshold=similarity_threshold,
        mean_adjacent_similarity=float(sims.mean()) if len(sims) else 1.0,
    )
    return shots


def drop_near_duplicate_shots(
    shot_embeddings: Embedding,
    *,
    similarity_threshold: float = 0.95,
) -> list[int]:
    """샷 대표 임베딩끼리 다시 비교해, 앞서 나온 샷과 거의 같은 샷을 버린다.

    브이로그는 같은 가게를 여러 번 돌아온다(먹는 컷 → 인테리어 → 다시 먹는 컷). 인접 비교만
    하면 이게 서로 다른 샷으로 남으므로, 전역 비교로 한 번 더 걸러낸다. 그리디 방식이라
    앞선 샷이 항상 살아남는다(시간 순서 보존).

    Args:
        shot_embeddings: (S, D) 샷 대표 임베딩.
        similarity_threshold: 이 값 이상이면 중복으로 판단.

    Returns:
        살아남은 샷의 인덱스(오름차순).
    """
    if len(shot_embeddings) == 0:
        return []

    kept: list[int] = [0]
    for index in range(1, len(shot_embeddings)):
        sims = shot_embeddings[kept] @ shot_embeddings[index]
        if float(sims.max()) < similarity_threshold:
            kept.append(index)

    logger.info(
        "duplicate_shots_dropped",
        shots_in=len(shot_embeddings),
        shots_out=len(kept),
        threshold=similarity_threshold,
    )
    return kept


def select_diverse_budget(
    shot_embeddings: Embedding,
    scores: list[float],
    *,
    budget: int,
) -> list[int]:
    """예산이 모자랄 때 **장면 다양성**을 기준으로 남길 샷을 고른다.

    점수 상위 K개를 남기는 방식에는 구조적인 문제가 있다. 점수는 "이 프레임이 읽기
    좋은가"만 보고 "이 장소가 이미 뽑혔는가"는 보지 않는다. 그래서 잘 찍힌 한 장소가
    슬롯 여러 개를 먹고, 조금 덜 찍힌 다른 장소는 **한 장도 남지 않는다.** 실측에서
    정상 노출 장소 한 곳이 이 경로로 통째로 사라졌고, 대신 다른 장소가 슬롯 3개를
    차지했다 (`benchmarks/validate_recall.py`).

    VLM 입력 예산의 목적은 "가장 예쁜 프레임 K장"이 아니라 "서로 다른 장소 K곳"이다.
    그래서 k-center greedy(최원점 우선)로 고른다. 가장 점수가 높은 샷에서 시작해,
    이미 고른 샷들과의 최대 유사도가 가장 낮은 샷을 반복해서 추가한다. 동점이면
    점수가 높은 쪽을 쓴다.

    Args:
        shot_embeddings: (S, D) L2 정규화된 샷 대표 임베딩.
        scores: 샷별 가독성 점수. 시작점과 동점 처리에만 쓴다.
        budget: 남길 샷 수.

    Returns:
        남길 샷의 인덱스(오름차순).
    """
    total = len(shot_embeddings)
    if total == 0:
        return []
    if budget >= total:
        return list(range(total))
    if budget <= 0:
        return []

    similarity = shot_embeddings @ shot_embeddings.T
    seed = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    chosen = [seed]
    # 이미 고른 집합과의 최대 유사도. 작을수록 새로운 장면이다.
    closeness = similarity[seed].astype(np.float64).copy()
    closeness[seed] = np.inf

    while len(chosen) < budget:
        # 최대 유사도가 가장 낮은 후보. 동점은 점수로 깬다.
        best = min(
            (index for index in range(total) if index not in chosen),
            key=lambda index: (closeness[index], -scores[index]),
        )
        chosen.append(best)
        closeness = np.minimum(closeness, similarity[best].astype(np.float64))
        closeness[best] = np.inf

    logger.info(
        "budget_applied",
        shots_in=total,
        shots_out=len(chosen),
        strategy="k-center-greedy",
    )
    return sorted(chosen)
