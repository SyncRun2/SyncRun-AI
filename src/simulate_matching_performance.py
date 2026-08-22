# ==============================
# 1. 기본 라이브러리
# ==============================

import random
from datetime import (
    datetime,
    timedelta,
)

import numpy as np

from matching_kmeans_engine import (
    GoalAwareKMeansGroupMatchingEngine,
)
from matching_kmeans_engine_mate_score import (
    MateScoreAwareKMeansGroupMatchingEngine,
    MatchingRequestWithMateScore,
)


# ==============================
# 2. 합성 데이터 생성
# ==============================

SIDO_SIGUNGU_DONGS = {
    ("서울", "강남구"): ["역삼동", "대치동", "삼성동", "청담동"],
    ("서울", "송파구"): ["잠실동", "문정동", "가락동"],
    ("서울", "마포구"): ["연남동", "합정동", "망원동"],
    ("부산", "해운대구"): ["우동", "중동", "좌동"],
}

GOALS = ["체중감량", "완주", "친목", "기록단축", "건강관리"]

DATES = [
    datetime(2026, 8, 20) + timedelta(days=offset)
    for offset in range(3)
]


def make_fake_request(
    request_id: int,
    rng: random.Random,
    single_pool: bool = False,
) -> MatchingRequestWithMateScore:
    if single_pool:
        sido, sigungu = ("서울", "강남구")
        date = DATES[0]
    else:
        sido, sigungu = rng.choice(
            list(SIDO_SIGUNGU_DONGS.keys())
        )
        date = rng.choice(DATES)

    dongs = rng.sample(
        SIDO_SIGUNGU_DONGS[(sido, sigungu)],
        k=rng.randint(1, 2),
    )

    start_hour = rng.randint(6, 20)

    start_at = date.replace(
        hour=start_hour,
        minute=rng.choice([0, 30]),
    )

    end_at = start_at + timedelta(
        hours=rng.choice([1, 2, 3])
    )

    pace_center = rng.randint(300, 420)

    pace_width = rng.randint(20, 60)

    group_min = rng.randint(2, 4)

    group_max = group_min + rng.randint(0, 2)

    # 당근 매너온도 방식: 대부분 42.195 근처, 일부는 매우 높거나 낮음
    mate_score = float(
        np.clip(
            rng.gauss(50, 20),
            0,
            100,
        )
    )

    return MatchingRequestWithMateScore(
        requestId=request_id,
        userId=request_id,
        regionSido=sido,
        regionSigungu=sigungu,
        regionDongs=dongs,
        runningGoal=rng.choice(GOALS),
        preferredStartAt=start_at,
        preferredEndAt=end_at,
        paceMinSec=pace_center - pace_width // 2,
        paceMaxSec=pace_center + pace_width // 2,
        groupSizeMin=group_min,
        groupSizeMax=group_max,
        mateScore=mate_score,
    )


# ==============================
# 3. 시뮬레이션 결과 집계 (여러 시드 평균)
# ==============================

def run_many_seeds(
    label: str,
    engine,
    seed_count: int,
    request_count: int,
    single_pool: bool = False,
):
    """
    시드를 여러 개 돌려서 평균 지표를 낸다.

    시드 1개짜리 결과는 그날 생성된 합성 데이터에 따라
    우연히 튈 수 있어서(특히 메이트 스코어 효과처럼
    가중치가 작은 지표), 발표용 수치는 항상 여러 번
    반복한 평균으로 뽑는다.
    """

    completion_rates = []
    group_scores_all = []
    all_avg_mate = []
    matched_avg_mate = []

    for seed in range(seed_count):
        rng = random.Random(seed)

        requests = [
            make_fake_request(
                request_id,
                rng,
                single_pool=single_pool,
            )
            for request_id in range(
                1,
                request_count + 1,
            )
        ]

        response = engine.recommend_groups(
            requests
        )

        matched_ids = {
            member.requestId
            for group in response.groups
            for member in group.members
        }

        completion_rates.append(
            len(matched_ids) / request_count
        )

        group_scores_all.extend(
            group.groupCompatibilityScore
            for group in response.groups
        )

        request_map = {
            request.requestId: request
            for request in requests
        }

        all_avg_mate.append(
            np.mean(
                [r.mateScore for r in requests]
            )
        )

        matched_scores = [
            request_map[member.requestId].mateScore
            for group in response.groups
            for member in group.members
        ]

        if matched_scores:
            matched_avg_mate.append(
                np.mean(matched_scores)
            )

    print(f"\n=== {label} (seed {seed_count}회 평균) ===")
    print(
        "평균 그룹 완성률: "
        f"{np.mean(completion_rates):.1%} "
        f"(표준편차 {np.std(completion_rates):.1%})"
    )

    if group_scores_all:
        print(
            "그룹 평균 궁합 점수: "
            f"{np.mean(group_scores_all):.2f}"
        )

    if matched_avg_mate:
        print(
            "전체 풀 평균 mateScore: "
            f"{np.mean(all_avg_mate):.2f} → "
            "매칭된 사용자 평균 mateScore: "
            f"{np.mean(matched_avg_mate):.2f}"
        )


if __name__ == "__main__":
    SEED_COUNT = 20

    REQUEST_COUNT = 300

    base_engine = GoalAwareKMeansGroupMatchingEngine()

    mate_engine = MateScoreAwareKMeansGroupMatchingEngine()

    run_many_seeds(
        "v2 (메이트 스코어 미반영) - 4지역x3일 분산",
        base_engine,
        SEED_COUNT,
        REQUEST_COUNT,
    )

    run_many_seeds(
        "v3 (메이트 스코어 15% 가산) - 4지역x3일 분산",
        mate_engine,
        SEED_COUNT,
        REQUEST_COUNT,
    )

    run_many_seeds(
        "v3 - 단일 지역/단일 날짜 집중 (Pool 밀도 비교용)",
        mate_engine,
        SEED_COUNT,
        REQUEST_COUNT,
        single_pool=True,
    )
