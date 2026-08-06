# ==============================
# 2. 목표 인식형 K-Means 매칭 엔진
# ==============================

import math

from datetime import datetime
from typing import Dict
from typing import List

import numpy as np

from sklearn.cluster import (
    KMeans,
)

from sklearn.feature_extraction import (
    DictVectorizer,
)

from schemas import (
    CandidateMatchRequest,
)

from schemas import (
    MatchMemberResponse,
)

from schemas import (
    MatchRequest,
)

from schemas import (
    MatchResponse,
)


class GoalAwareKMeansMatchingEngine:
    """
    같은 시·군·구의 매칭 대기 사용자를 대상으로
    복수 선호 동, 시간, 페이스, 러닝 목표를 벡터화한다.

    K-Means 군집화와 사용자 궁합 점수를 결합하여
    최종 러닝 그룹을 추천한다.
    """

    def __init__(
        self,
    ):
        """
        K-Means 특징 가중치와
        최종 궁합 점수 가중치를 설정한다.
        """

        # ==============================
        # K-Means 특징 가중치
        # ==============================

        self.feature_weights = {
            "dong": 1.50,
            "goal": 1.20,
            "time": 1.50,
            "pace": 1.40,
        }

        # ==============================
        # 최종 궁합 점수 가중치
        # ==============================

        self.score_weights = {
            "time": 0.35,
            "pace": 0.30,
            "dong": 0.20,
            "goal": 0.15,
        }


    # ==============================
    # 2-1. 최종 러닝 그룹 추천
    # ==============================

    def recommend_group(
        self,
        target_request: MatchRequest,
        pending_requests: List[
            CandidateMatchRequest
        ],
    ) -> MatchResponse:
        """
        현재 요청자를 기준으로
        가장 적합한 러닝 그룹 구성원을 추천한다.
        """

        # ==============================
        # 1차 후보 필터링
        # ==============================

        candidate_requests = (
            self._filter_candidates(
                target_request=(
                    target_request
                ),
                pending_requests=(
                    pending_requests
                ),
            )
        )

        # ==============================
        # 후보가 없는 경우
        # ==============================

        if len(
            candidate_requests
        ) == 0:
            return MatchResponse(
                algorithm=(
                    "GoalAwareKMeansMatchingEngine-v1"
                ),

                candidateCount=0,
                clusterCount=0,

                requestedGroupSize=(
                    target_request.groupSize
                ),

                finalGroupSize=1,

                groupMembers=[],

                message=(
                    "같은 구에서 날짜와 시간대가 "
                    "겹치는 후보가 없습니다."
                ),
            )

        # ==============================
        # 요청자 + 후보자 결합
        # ==============================

        all_requests = [
            target_request,
            *candidate_requests,
        ]

        # ==============================
        # 특징 데이터 생성
        # ==============================

        feature_records = [
            self._create_feature_record(
                request=request,
            )
            for request in all_requests
        ]

        # ==============================
        # Dict 특징을 숫자 행렬로 변환
        # ==============================

        vectorizer = DictVectorizer(
            sparse=False,
        )

        feature_matrix = (
            vectorizer.fit_transform(
                feature_records
            )
        )

        # ==============================
        # 클러스터 개수 계산
        # ==============================

        cluster_count = (
            self._calculate_cluster_count(
                feature_matrix=(
                    feature_matrix
                ),
                group_size=(
                    target_request.groupSize
                ),
            )
        )

        # ==============================
        # K-Means 모델 생성
        # ==============================

        kmeans_model = KMeans(
            n_clusters=(
                cluster_count
            ),
            init="k-means++",
            n_init=10,
            random_state=42,
        )

        # ==============================
        # K-Means 학습 및 군집 예측
        # ==============================

        cluster_labels = (
            kmeans_model.fit_predict(
                feature_matrix
            )
        )

        # 첫 번째 데이터는 현재 요청자
        target_cluster_id = int(
            cluster_labels[0]
        )

        ranked_candidates = []

        # ==============================
        # 후보자별 궁합 계산
        # ==============================

        for candidate_index, candidate in enumerate(
            candidate_requests,
            start=1,
        ):
            candidate_cluster_id = int(
                cluster_labels[
                    candidate_index
                ]
            )

            same_cluster = (
                candidate_cluster_id
                == target_cluster_id
            )

            compatibility_score = (
                self._calculate_compatibility_score(
                    target=(
                        target_request
                    ),
                    candidate=(
                        candidate
                    ),
                )
            )

            common_dongs = (
                self._get_common_dongs(
                    target=(
                        target_request
                    ),
                    candidate=(
                        candidate
                    ),
                )
            )

            reasons = (
                self._create_match_reasons(
                    target=(
                        target_request
                    ),
                    candidate=(
                        candidate
                    ),
                )
            )

            ranked_candidates.append(
                {
                    "candidate": (
                        candidate
                    ),

                    "clusterId": (
                        candidate_cluster_id
                    ),

                    "sameCluster": (
                        same_cluster
                    ),

                    "score": (
                        compatibility_score
                    ),

                    "commonDongs": (
                        common_dongs
                    ),

                    "reasons": (
                        reasons
                    ),
                }
            )

        # ==============================
        # 추천 우선순위 정렬
        #
        # 1. 요청자와 같은 클러스터
        # 2. 궁합 점수가 높은 사용자
        # ==============================

        ranked_candidates.sort(
            key=lambda result: (
                result[
                    "sameCluster"
                ],
                result[
                    "score"
                ],
            ),
            reverse=True,
        )

        # groupSize에는 요청자 본인도 포함
        required_member_count = max(
            target_request.groupSize - 1,
            0,
        )

        selected_candidates = (
            ranked_candidates[
                :required_member_count
            ]
        )

        # ==============================
        # 응답 데이터 생성
        # ==============================

        group_members = [
            MatchMemberResponse(
                requestId=(
                    result[
                        "candidate"
                    ].requestId
                ),

                userId=(
                    result[
                        "candidate"
                    ].userId
                ),

                nickname=(
                    result[
                        "candidate"
                    ].nickname
                ),

                regionDongs=(
                    result[
                        "candidate"
                    ].regionDongs
                ),

                commonDongs=(
                    result[
                        "commonDongs"
                    ]
                ),

                runningGoal=(
                    result[
                        "candidate"
                    ].runningGoal
                ),

                paceMinSec=(
                    result[
                        "candidate"
                    ].paceMinSec
                ),

                paceMaxSec=(
                    result[
                        "candidate"
                    ].paceMaxSec
                ),

                preferredStartAt=(
                    result[
                        "candidate"
                    ].preferredStartAt
                ),

                preferredEndAt=(
                    result[
                        "candidate"
                    ].preferredEndAt
                ),

                clusterId=(
                    result[
                        "clusterId"
                    ]
                ),

                sameCluster=(
                    result[
                        "sameCluster"
                    ]
                ),

                compatibilityScore=(
                    result[
                        "score"
                    ]
                ),

                reasons=(
                    result[
                        "reasons"
                    ]
                ),
            )
            for result in selected_candidates
        ]

        final_group_size = (
            1
            + len(
                group_members
            )
        )

        # ==============================
        # 결과 메시지 생성
        # ==============================

        if (
            final_group_size
            >= target_request.groupSize
        ):
            message = (
                f"{target_request.groupSize}명 "
                "러닝 그룹 구성이 완료되었습니다."
            )

        else:
            message = (
                "조건을 만족하는 후보가 부족하여 "
                f"{final_group_size}명까지만 "
                "추천되었습니다."
            )

        return MatchResponse(
            algorithm=(
                "GoalAwareKMeansMatchingEngine-v1"
            ),

            candidateCount=len(
                candidate_requests
            ),

            clusterCount=(
                cluster_count
            ),

            requestedGroupSize=(
                target_request.groupSize
            ),

            finalGroupSize=(
                final_group_size
            ),

            groupMembers=(
                group_members
            ),

            message=(
                message
            ),
        )


    # ==============================
    # 2-2. 1차 후보 필터링
    # ==============================

    def _filter_candidates(
        self,
        target_request: MatchRequest,
        pending_requests: List[
            CandidateMatchRequest
        ],
    ) -> List[
        CandidateMatchRequest
    ]:
        """
        다음 조건을 만족하는 사용자만 후보로 구성한다.

        1. 같은 시·도
        2. 같은 시·군·구
        3. 같은 날짜
        4. 선호 시간 구간이 겹침

        선호 동은 후보 제외 조건으로 사용하지 않는다.
        선호 동은 K-Means와 궁합 점수에 반영한다.
        """

        filtered_candidates = []

        for candidate in pending_requests:
            same_sido = (
                candidate.regionSido
                == target_request.regionSido
            )

            same_sigungu = (
                candidate.regionSigungu
                == target_request.regionSigungu
            )

            same_date = (
                candidate.preferredStartAt.date()
                == target_request.preferredStartAt.date()
            )

            overlap_seconds = (
                self._calculate_datetime_overlap_seconds(
                    first_start=(
                        target_request.preferredStartAt
                    ),

                    first_end=(
                        target_request.preferredEndAt
                    ),

                    second_start=(
                        candidate.preferredStartAt
                    ),

                    second_end=(
                        candidate.preferredEndAt
                    ),
                )
            )

            has_time_overlap = (
                overlap_seconds > 0
            )

            if (
                same_sido
                and same_sigungu
                and same_date
                and has_time_overlap
            ):
                filtered_candidates.append(
                    candidate
                )

        return filtered_candidates


    # ==============================
    # 2-3. AI 입력 특징 벡터 생성
    # ==============================

    def _create_feature_record(
        self,
        request: MatchRequest,
    ) -> Dict[
        str,
        float
    ]:
        """
        복수 선호 동, 목표, 시간, 페이스를
        K-Means 입력용 숫자 특징으로 변환한다.
        """

        start_at = (
            request.preferredStartAt
        )

        end_at = (
            request.preferredEndAt
        )

        # ==============================
        # 선호 시간의 중앙값 계산
        # ==============================

        midpoint_at = (
            start_at
            + (
                end_at
                - start_at
            )
            / 2
        )

        midpoint_minute = (
            midpoint_at.hour
            * 60
            + midpoint_at.minute
            + midpoint_at.second
            / 60
        )

        # ==============================
        # 시간을 순환형 sin/cos 특징으로 변환
        # ==============================

        time_angle = (
            2
            * math.pi
            * midpoint_minute
            / 1440
        )

        duration_minutes = (
            end_at
            - start_at
        ).total_seconds() / 60

        # ==============================
        # 페이스 중앙값 및 범위 계산
        # ==============================

        pace_center = (
            request.paceMinSec
            + request.paceMaxSec
        ) / 2

        pace_width = (
            request.paceMaxSec
            - request.paceMinSec
        )

        # ==============================
        # 기본 특징 생성
        # ==============================

        feature_record = {
            (
                "goal="
                + request.runningGoal
            ): self.feature_weights[
                "goal"
            ],

            "time_sin": (
                math.sin(
                    time_angle
                )
                * self.feature_weights[
                    "time"
                ]
            ),

            "time_cos": (
                math.cos(
                    time_angle
                )
                * self.feature_weights[
                    "time"
                ]
            ),

            "time_duration": (
                duration_minutes
                / 240
                * 0.40
            ),

            "pace_center": (
                pace_center
                / 600
                * self.feature_weights[
                    "pace"
                ]
            ),

            "pace_width": (
                pace_width
                / 180
                * 0.40
            ),
        }

        # ==============================
        # 복수 선택 동을 Multi-Hot 특징으로 추가
        # ==============================

        selected_dong_count = max(
            len(
                request.regionDongs
            ),
            1,
        )

        # 여러 동을 선택해도 동 특징의 전체 크기가
        # 과도하게 증가하지 않도록 정규화한다.
        dong_feature_weight = (
            self.feature_weights[
                "dong"
            ]
            / math.sqrt(
                selected_dong_count
            )
        )

        for dong in request.regionDongs:
            feature_record[
                "dong="
                + dong
            ] = dong_feature_weight

        return feature_record


    # ==============================
    # 2-4. 클러스터 개수 계산
    # ==============================

    def _calculate_cluster_count(
        self,
        feature_matrix: np.ndarray,
        group_size: int,
    ) -> int:
        """
        전체 사용자 수와 희망 그룹 크기를 이용해
        K-Means 클러스터 개수를 계산한다.

        동일한 특징 벡터가 존재하는 경우를 고려하여
        고유 특징 벡터 개수를 초과하지 않도록 한다.
        """

        sample_count = int(
            feature_matrix.shape[0]
        )

        estimated_cluster_count = (
            math.ceil(
                sample_count
                / group_size
            )
        )

        unique_sample_count = len(
            np.unique(
                feature_matrix,
                axis=0,
            )
        )

        cluster_count = max(
            1,
            estimated_cluster_count,
        )

        cluster_count = min(
            cluster_count,
            sample_count,
            unique_sample_count,
        )

        return int(
            cluster_count
        )


    # ==============================
    # 2-5. 공통 선호 동 계산
    # ==============================

    def _get_common_dongs(
        self,
        target: MatchRequest,
        candidate: CandidateMatchRequest,
    ) -> List[
        str
    ]:
        """
        두 사용자가 공통으로 선택한 동을 반환한다.

        반환 순서는 현재 요청자가 입력한
        선호 동 순서를 유지한다.
        """

        candidate_dong_set = set(
            candidate.regionDongs
        )

        common_dongs = [
            dong
            for dong in target.regionDongs
            if dong in candidate_dong_set
        ]

        return common_dongs


    # ==============================
    # 2-6. 선호 동 유사도 계산
    # ==============================

    def _calculate_dong_similarity(
        self,
        target: MatchRequest,
        candidate: CandidateMatchRequest,
    ) -> float:
        """
        복수 선호 동의 Jaccard 유사도를 계산한다.

        유사도 =
        공통 동 개수 / 전체 고유 동 개수
        """

        target_dongs = set(
            target.regionDongs
        )

        candidate_dongs = set(
            candidate.regionDongs
        )

        union_dongs = (
            target_dongs
            | candidate_dongs
        )

        if len(
            union_dongs
        ) == 0:
            return 0.0

        common_dongs = (
            target_dongs
            & candidate_dongs
        )

        similarity = (
            len(
                common_dongs
            )
            / len(
                union_dongs
            )
        )

        return float(
            similarity
        )


    # ==============================
    # 2-7. 최종 궁합 점수 계산
    # ==============================

    def _calculate_compatibility_score(
        self,
        target: MatchRequest,
        candidate: CandidateMatchRequest,
    ) -> float:
        """
        시간, 페이스, 선호 동, 목표를 조합하여
        0~100점의 최종 궁합 점수를 계산한다.
        """

        time_score = (
            self._calculate_time_similarity(
                target=(
                    target
                ),
                candidate=(
                    candidate
                ),
            )
        )

        pace_score = (
            self._calculate_pace_similarity(
                target=(
                    target
                ),
                candidate=(
                    candidate
                ),
            )
        )

        dong_score = (
            self._calculate_dong_similarity(
                target=(
                    target
                ),
                candidate=(
                    candidate
                ),
            )
        )

        if (
            target.runningGoal
            == candidate.runningGoal
        ):
            goal_score = 1.0

        else:
            goal_score = 0.20

        total_score = (
            time_score
            * self.score_weights[
                "time"
            ]

            + pace_score
            * self.score_weights[
                "pace"
            ]

            + dong_score
            * self.score_weights[
                "dong"
            ]

            + goal_score
            * self.score_weights[
                "goal"
            ]
        )

        return round(
            total_score * 100,
            2,
        )


    # ==============================
    # 2-8. 시간 유사도 계산
    # ==============================

    def _calculate_time_similarity(
        self,
        target: MatchRequest,
        candidate: CandidateMatchRequest,
    ) -> float:
        """
        두 선호 시간 구간의 겹치는 비율을 계산한다.

        더 짧은 시간 구간을 기준으로
        0~1 사이의 유사도를 계산한다.
        """

        overlap_seconds = (
            self._calculate_datetime_overlap_seconds(
                first_start=(
                    target.preferredStartAt
                ),

                first_end=(
                    target.preferredEndAt
                ),

                second_start=(
                    candidate.preferredStartAt
                ),

                second_end=(
                    candidate.preferredEndAt
                ),
            )
        )

        target_duration = (
            target.preferredEndAt
            - target.preferredStartAt
        ).total_seconds()

        candidate_duration = (
            candidate.preferredEndAt
            - candidate.preferredStartAt
        ).total_seconds()

        shorter_duration = min(
            target_duration,
            candidate_duration,
        )

        if (
            shorter_duration
            <= 0
        ):
            return 0.0

        similarity = (
            overlap_seconds
            / shorter_duration
        )

        return float(
            np.clip(
                similarity,
                0.0,
                1.0,
            )
        )


    # ==============================
    # 2-9. 페이스 유사도 계산
    # ==============================

    def _calculate_pace_similarity(
        self,
        target: MatchRequest,
        candidate: CandidateMatchRequest,
    ) -> float:
        """
        페이스 범위 교집합과 중앙값 차이를 이용해
        0~1 사이의 페이스 유사도를 계산한다.
        """

        # ==============================
        # 페이스 범위 교집합
        # ==============================

        overlap_start = max(
            target.paceMinSec,
            candidate.paceMinSec,
        )

        overlap_end = min(
            target.paceMaxSec,
            candidate.paceMaxSec,
        )

        overlap_width = max(
            0,
            overlap_end
            - overlap_start,
        )

        # ==============================
        # 페이스 범위 합집합
        # ==============================

        union_start = min(
            target.paceMinSec,
            candidate.paceMinSec,
        )

        union_end = max(
            target.paceMaxSec,
            candidate.paceMaxSec,
        )

        union_width = max(
            0,
            union_end
            - union_start,
        )

        # ==============================
        # 페이스 중앙값 차이
        # ==============================

        target_center = (
            target.paceMinSec
            + target.paceMaxSec
        ) / 2

        candidate_center = (
            candidate.paceMinSec
            + candidate.paceMaxSec
        ) / 2

        center_difference = abs(
            target_center
            - candidate_center
        )

        # 중앙 페이스가 180초/km 이상 차이 나면
        # 중앙값 기반 유사도를 0으로 처리한다.
        center_similarity = (
            1.0
            - center_difference
            / 180
        )

        center_similarity = float(
            np.clip(
                center_similarity,
                0.0,
                1.0,
            )
        )

        # ==============================
        # 범위 교집합 유사도
        # ==============================

        if (
            union_width > 0
        ):
            overlap_similarity = (
                overlap_width
                / union_width
            )

        else:
            overlap_similarity = (
                1.0
                if center_difference == 0
                else 0.0
            )

        # 중앙값 차이 70%
        # 범위 교집합 30%
        similarity = (
            center_similarity
            * 0.70

            + overlap_similarity
            * 0.30
        )

        return float(
            np.clip(
                similarity,
                0.0,
                1.0,
            )
        )


    # ==============================
    # 2-10. 추천 사유 생성
    # ==============================

    def _create_match_reasons(
        self,
        target: MatchRequest,
        candidate: CandidateMatchRequest,
    ) -> List[
        str
    ]:
        """
        추천 결과와 함께 사용자에게 제공할
        추천 근거 문구를 생성한다.
        """

        reasons = []

        # ==============================
        # 공통 선호 동
        # ==============================

        common_dongs = (
            self._get_common_dongs(
                target=(
                    target
                ),
                candidate=(
                    candidate
                ),
            )
        )

        if len(
            common_dongs
        ) > 0:
            reasons.append(
                "공통 선호 동: "
                + ", ".join(
                    common_dongs
                )
            )

        # ==============================
        # 시간 겹침
        # ==============================

        overlap_seconds = (
            self._calculate_datetime_overlap_seconds(
                first_start=(
                    target.preferredStartAt
                ),

                first_end=(
                    target.preferredEndAt
                ),

                second_start=(
                    candidate.preferredStartAt
                ),

                second_end=(
                    candidate.preferredEndAt
                ),
            )
        )

        overlap_minutes = int(
            overlap_seconds
            // 60
        )

        if (
            overlap_minutes > 0
        ):
            reasons.append(
                f"선호 시간이 "
                f"{overlap_minutes}분 겹칩니다."
            )

        # ==============================
        # 페이스 범위 겹침
        # ==============================

        pace_overlap_start = max(
            target.paceMinSec,
            candidate.paceMinSec,
        )

        pace_overlap_end = min(
            target.paceMaxSec,
            candidate.paceMaxSec,
        )

        if (
            pace_overlap_start
            < pace_overlap_end
        ):
            reasons.append(
                f"페이스 범위가 "
                f"{pace_overlap_start}~"
                f"{pace_overlap_end}초/km로 "
                "겹칩니다."
            )

        else:
            target_center = (
                target.paceMinSec
                + target.paceMaxSec
            ) / 2

            candidate_center = (
                candidate.paceMinSec
                + candidate.paceMaxSec
            ) / 2

            pace_center_difference = abs(
                target_center
                - candidate_center
            )

            if (
                pace_center_difference
                <= 30
            ):
                reasons.append(
                    "평균 페이스 차이가 "
                    f"{int(pace_center_difference)}초/km입니다."
                )

        # ==============================
        # 러닝 목표 일치
        # ==============================

        if (
            target.runningGoal
            == candidate.runningGoal
        ):
            reasons.append(
                "러닝 목표가 같습니다."
            )

        if len(
            reasons
        ) == 0:
            reasons.append(
                "같은 지역의 유사 사용자입니다."
            )

        return reasons


    # ==============================
    # 2-11. 시간 구간 교집합 계산
    # ==============================

    def _calculate_datetime_overlap_seconds(
        self,
        first_start: datetime,
        first_end: datetime,
        second_start: datetime,
        second_end: datetime,
    ) -> float:
        """
        두 시간 구간이 겹치는 시간을
        초 단위로 반환한다.
        """

        overlap_start = max(
            first_start,
            second_start,
        )

        overlap_end = min(
            first_end,
            second_end,
        )

        overlap_seconds = (
            overlap_end
            - overlap_start
        ).total_seconds()

        return max(
            0.0,
            overlap_seconds,
        )