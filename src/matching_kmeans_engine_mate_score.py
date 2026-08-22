# ==============================
# 1. 기본 라이브러리
# ==============================

import numpy as np

from pydantic import (
    Field,
)


# ==============================
# 2. 기존 엔진 / Schema
# ==============================

from matching_kmeans_engine import (
    GoalAwareKMeansGroupMatchingEngine,
)

from schemas import (
    MatchingRequest,
)


# ==============================
# 3. 메이트 스코어 포함 매칭 요청 Schema
# ==============================

class MatchingRequestWithMateScore(
    MatchingRequest
):
    """
    기존 MatchingRequest에
    메이트 스코어(당근 매너온도 방식)를 추가한 요청.

    0~100 스코어,
    신규 유저 기본값은 42.195
    (마라톤 풀코스 거리에서 착안).
    """

    mateScore: float = Field(
        default=42.195,
        ge=0,
        le=100,
    )


# ==============================
# 4. 메이트 스코어 반영 매칭 엔진
# ==============================

class MateScoreAwareKMeansGroupMatchingEngine(
    GoalAwareKMeansGroupMatchingEngine
):
    """
    GoalAwareKMeansGroupMatchingEngine을 그대로 상속해

    - Pool 생성
    - 페이스 기준 K-Means 클러스터링
    - Seed 선택 / 그룹 크기 후보 결정
    - 그룹 전체 공통 시간 검사
    - 조합 탐색으로 최적 그룹 채택

    로직은 전혀 건드리지 않는다.

    오직 사용자 Pair 궁합 점수(_calculate_compatibility_score)에
    메이트 스코어를 '가산점'으로 추가 반영한다.

    정원 초과 상황(예: 5명이 적합한데 정원 4명)에서
    하드 컷을 하지 않는다. 대신 _find_best_group의
    조합 탐색이 그룹 평균 점수가 가장 높은 4명 조합을
    고르게 되는데, 메이트 스코어가 그 평균 점수에
    녹아 있으므로 결과적으로 매너 좋은 사람이
    포함된 조합이 자연스럽게 우선된다.
    """


    def __init__(
        self,
    ):
        super().__init__()

        # ==============================
        # 4-1. Compatibility Weight
        # (메이트 스코어 15% 추가,
        # 나머지는 비율 유지하며 축소)
        # ==============================

        self.score_weights = {
            "time": 0.30,
            "pace": 0.25,
            "dong": 0.17,
            "goal": 0.13,
            "mate": 0.15,
        }


        # ==============================
        # 4-2. 알고리즘 버전
        # ==============================

        self.algorithm_name = (
            "MateScoreAwareKMeans"
            "GroupMatchingEngine-v3"
        )


    # ==============================
    # 5. 사용자 Pair 궁합
    # (메이트 스코어 가산점 포함)
    # ==============================

    def _calculate_compatibility_score(
        self,
        request_a: MatchingRequestWithMateScore,
        request_b: MatchingRequestWithMateScore,
    ) -> float:
        """
        시간       30%
        페이스     25%
        선호 동    17%
        러닝 목표  13%
        메이트 스코어 15%
        """

        # ==============================
        # 시간
        # ==============================

        time_score = (
            self
            ._calculate_time_similarity(
                request_a,
                request_b,
            )
        )


        # ==============================
        # 페이스
        # ==============================

        pace_score = (
            self
            ._calculate_pace_similarity(
                request_a,
                request_b,
            )
        )


        # ==============================
        # 복수 동 Jaccard 유사도
        # ==============================

        dong_score = (
            self
            ._calculate_dong_similarity(
                request_a,
                request_b,
            )
        )


        # ==============================
        # 러닝 목표
        # ==============================

        goal_score = (
            1.0
            if (
                request_a.runningGoal
                == request_b.runningGoal
            )
            else 0.20
        )


        # ==============================
        # 메이트 스코어 가산점
        # ==============================

        mate_score = (
            self
            ._calculate_mate_score_bonus(
                request_a,
                request_b,
            )
        )


        # ==============================
        # 최종 점수
        # ==============================

        total_score = (
            time_score
            * self.score_weights[
                "time"
            ]
            +
            pace_score
            * self.score_weights[
                "pace"
            ]
            +
            dong_score
            * self.score_weights[
                "dong"
            ]
            +
            goal_score
            * self.score_weights[
                "goal"
            ]
            +
            mate_score
            * self.score_weights[
                "mate"
            ]
        )


        return round(
            total_score
            * 100,
            2,
        )


    # ==============================
    # 6. 메이트 스코어 가산점 계산
    # ==============================

    @staticmethod
    def _calculate_mate_score_bonus(
        request_a: MatchingRequestWithMateScore,
        request_b: MatchingRequestWithMateScore,
    ) -> float:
        """
        메이트 스코어(0~100)를 0~1로 정규화한 뒤
        두 사용자의 평균을 가산점으로 사용한다.

        두 값의 '유사도'가 아니라
        '둘 다 매너 좋은 메이트일수록' 유리하도록
        평균을 사용한다.

        한쪽이 매우 낮으면 평균이 함께 낮아지므로
        저평판 유저가 고평판 유저 뒤에 숨어
        가산점만 챙기는 상황을 막는다.
        """

        normalized_a = float(
            np.clip(
                request_a.mateScore
                / 100,
                0.0,
                1.0,
            )
        )


        normalized_b = float(
            np.clip(
                request_b.mateScore
                / 100,
                0.0,
                1.0,
            )
        )


        return (
            normalized_a
            + normalized_b
        ) / 2
