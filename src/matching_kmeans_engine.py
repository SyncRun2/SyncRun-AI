# ==============================
# 1. 기본 라이브러리
# ==============================

import math

from itertools import (
    combinations,
)

from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)


# ==============================
# 2. 데이터 처리
# ==============================

import numpy as np

from sklearn.cluster import (
    KMeans,
)

from sklearn.feature_extraction import (
    DictVectorizer,
)

from sklearn.metrics import (
    silhouette_score,
)


# ==============================
# 3. Schema
# ==============================

from schemas import (
    GroupMemberResponse,
    MatchBatchResponse,
    MatchingRequest,
)


# ==============================
# 4. AI 그룹 매칭 엔진
# ==============================

class GoalAwareKMeansGroupMatchingEngine:
    """
    전체 MATCHING 요청을 이용해

    1. 지역 / 날짜 Pool 생성
    2. K-Means 군집화
    3. 사용자 궁합 계산
    4. 그룹 최소 / 최대 인원 검사
    5. 그룹 전체 공통 시간 검사
    6. 최종 그룹 구성

    을 수행한다.
    """


    def __init__(
        self,
    ):
        # ==============================
        # 4-2. Compatibility Weight
        # ==============================

        self.score_weights = {
            "time": 0.35,
            "pace": 0.30,
            "dong": 0.20,
            "goal": 0.15,
        }


        # ==============================
        # 4-3. 알고리즘 버전
        # ==============================

        self.algorithm_name = (
            "GoalAwareKMeans"
            "GroupMatchingEngine-v2"
        )


    # ==============================
    # 5. 전체 그룹 매칭
    # ==============================

    def recommend_groups(
        self,
        requests: List[
            MatchingRequest
        ],
    ) -> MatchBatchResponse:
        """
        현재 MATCHING 요청 전체에서
        완성 가능한 그룹들을 생성한다.
        """

        # ==============================
        # 5-1. Pool 생성
        # ==============================

        pools = (
            self._create_matching_pools(
                requests
            )
        )


        all_groups = []

        group_no = 1


        # ==============================
        # 5-2. Pool별 매칭
        # ==============================

        for pool_requests in (
            pools.values()
        ):
            pool_groups = (
                self._match_pool(
                    pool_requests
                )
            )


            for group in (
                pool_groups
            ):
                group[
                    "groupNo"
                ] = group_no

                all_groups.append(
                    group
                )

                group_no += 1


        # ==============================
        # 5-3. 최종 결과
        # ==============================

        return MatchBatchResponse(
            algorithm=(
                self.algorithm_name
            ),
            groupCount=len(
                all_groups
            ),
            groups=(
                all_groups
            ),
        )


    # ==============================
    # 6. Matching Pool 생성
    # ==============================

    def _create_matching_pools(
        self,
        requests: List[
            MatchingRequest
        ],
    ) -> Dict[
        Tuple,
        List[
            MatchingRequest
        ],
    ]:
        """
        동일한

        - 시/도
        - 시/군/구
        - 러닝 날짜

        단위로 Pool을 만든다.
        """

        pools = {}


        for request in requests:
            pool_key = (
                request.regionSido,
                request.regionSigungu,
                request
                .preferredStartAt
                .date(),
            )


            if (
                pool_key
                not in pools
            ):
                pools[
                    pool_key
                ] = []


            pools[
                pool_key
            ].append(
                request
            )


        return pools


    # ==============================
    # 7. Pool 내부 그룹 생성
    # ==============================

    def _match_pool(
        self,
        requests: List[
            MatchingRequest
        ],
    ) -> List[
        Dict
    ]:
        """
        하나의 Pool을
        페이스 기준 K-Means Cluster로
        먼저 나눈 뒤,

        각 Cluster 내부에서만
        그룹을 생성한다.

        Cluster 경계를 벗어난 매칭은
        시도하지 않는다. 이번 라운드에
        묶이지 못한 사람은 matching 상태로
        남아 다음 요청 도착 시
        Pool 전체가 재구성되며
        다시 시도된다.
        """

        if not requests:
            return []


        # ==============================
        # 7-1. 페이스 기준 K-Means
        # ==============================

        cluster_map = (
            self._run_kmeans(
                requests
            )
        )


        # ==============================
        # 7-2. Cluster 단위로 분할
        # ==============================

        clustered_requests: Dict[
            int,
            List[
                MatchingRequest
            ],
        ] = {}


        for request in requests:
            cluster_label = (
                cluster_map[
                    request.requestId
                ]
            )

            clustered_requests.setdefault(
                cluster_label,
                [],
            ).append(
                request
            )


        # ==============================
        # 7-3. Cluster별 매칭
        # ==============================

        groups = []


        for cluster_requests in (
            clustered_requests
            .values()
        ):
            groups.extend(
                self._match_cluster(
                    cluster_requests
                )
            )


        return groups


    # ==============================
    # 7-A. Cluster 내부 그룹 생성
    # ==============================

    def _match_cluster(
        self,
        requests: List[
            MatchingRequest
        ],
    ) -> List[
        Dict
    ]:
        """
        하나의 Cluster 안에서
        가능한 그룹들을 생성한다.
        """

        # ==============================
        # 아직 그룹에 들어가지 않은 요청
        # ==============================

        remaining = list(
            requests
        )


        groups = []


        attempted_request_ids = set()


        # ==============================
        # 반복 매칭
        # ==============================

        while remaining:
            available_seeds = [
                request
                for request
                in remaining
                if request.requestId
                not in attempted_request_ids
            ]


            if not available_seeds:
                break


            # ==============================
            # 가장 조건이 제한적인
            # 사용자부터 매칭
            # ==============================

            seed = (
                self._select_seed(
                    available_seeds
                )
            )


            # ==============================
            # Seed 기준 최적 그룹 탐색
            # ==============================

            best_group = (
                self._find_best_group(
                    seed=seed,
                    remaining=remaining,
                )
            )


            # ==============================
            # 현재 Seed로 그룹 생성 실패
            # ==============================

            if (
                best_group
                is None
            ):
                attempted_request_ids.add(
                    seed.requestId
                )

                continue


            # ==============================
            # 그룹 생성 성공
            # ==============================

            group_response = (
                self._create_group_response(
                    best_group
                )
            )


            groups.append(
                group_response
            )


            # ==============================
            # 그룹에 들어간 Request ID
            # ==============================

            matched_request_ids = {
                member.requestId
                for member
                in best_group
            }


            # ==============================
            # 매칭 완료된 사람 제거
            # ==============================

            remaining = [
                request
                for request
                in remaining
                if request.requestId
                not in matched_request_ids
            ]


        return groups


    # ==============================
    # 8. K-Means 수행
    # ==============================

    def _run_kmeans(
        self,
        requests: List[
            MatchingRequest
        ],
    ) -> Dict[
        int,
        int,
    ]:
        """
        사용자 특징을 Vector로 변환하고
        K-Means를 수행한다.
        """

        if len(
            requests
        ) == 1:
            return {
                requests[
                    0
                ].requestId: 0
            }


        # ==============================
        # 8-1. Feature 생성
        # ==============================

        feature_records = [
            self._create_feature_record(
                request
            )
            for request
            in requests
        ]


        # ==============================
        # 8-2. Dict → Vector
        # ==============================

        vectorizer = (
            DictVectorizer(
                sparse=False
            )
        )


        feature_matrix = (
            vectorizer
            .fit_transform(
                feature_records
            )
        )


        # ==============================
        # 8-3. 실루엣 스코어 기반
        # 최적 Cluster 라벨링
        # ==============================

        labels = (
            self._cluster_by_silhouette(
                requests=(
                    requests
                ),
                feature_matrix=(
                    feature_matrix
                ),
            )
        )


        return {
            request.requestId: int(
                label
            )
            for request, label
            in zip(
                requests,
                labels,
            )
        }


    # ==============================
    # 9. Feature 생성
    # ==============================

    def _create_feature_record(
        self,
        request: MatchingRequest,
    ) -> Dict[
        str,
        float,
    ]:
        """
        페이스 중앙값 / 범위를
        Feature Vector로 변환한다.

        동 / 목표 / 시간은
        Pool 필터, 하드 제약(공통 시간대),
        Pairwise 궁합 점수에서 이미
        처리되므로 K-Means에는 넣지 않는다.

        페이스는 다른 곳에서 강제되지 않는
        유일한 연속값이라 군집화 축으로 삼는다.
        """

        pace_center = (
            request.paceMinSec
            + request.paceMaxSec
        ) / 2


        pace_width = (
            request.paceMaxSec
            - request.paceMinSec
        )


        return {
            "pace_center": (
                pace_center
                / 600
            ),

            "pace_width": (
                pace_width
                / 180
            ),
        }


    # ==============================
    # 10. 실루엣 스코어 기반
    # Cluster 라벨링
    # ==============================

    def _cluster_by_silhouette(
        self,
        requests: List[
            MatchingRequest
        ],
        feature_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        평균 희망 그룹 크기로 추정한 k
        주변을 후보로 놓고,
        실루엣 스코어가 가장 높은 k의
        Cluster 라벨을 반환한다.
        """

        sample_count = len(
            requests
        )


        # ==============================
        # 10-1. 평균 희망 그룹 크기
        # 기준 k 추정 (탐색 중심점)
        # ==============================

        average_group_size = (
            sum(
                (
                    request.groupSizeMin
                    + request.groupSizeMax
                )
                / 2
                for request
                in requests
            )
            / sample_count
        )


        estimated_cluster_count = (
            math.ceil(
                sample_count
                / max(
                    average_group_size,
                    1,
                )
            )
        )


        # ==============================
        # 10-2. 실제로 가능한
        # 최대 Cluster 개수
        # ==============================

        unique_vector_count = (
            np.unique(
                feature_matrix,
                axis=0,
            )
            .shape[
                0
            ]
        )


        max_feasible_k = min(
            sample_count - 1,
            unique_vector_count,
        )


        if (
            max_feasible_k
            < 2
        ):
            return np.zeros(
                sample_count,
                dtype=int,
            )


        # ==============================
        # 10-3. 추정 k 주변에서
        # 실루엣 스코어 탐색
        # ==============================

        search_radius = 2

        candidate_k_values = [
            k
            for k in range(
                max(
                    2,
                    estimated_cluster_count
                    - search_radius,
                ),
                min(
                    max_feasible_k,
                    estimated_cluster_count
                    + search_radius,
                )
                + 1,
            )
        ]

        if not candidate_k_values:
            candidate_k_values = [
                max(
                    2,
                    min(
                        max_feasible_k,
                        estimated_cluster_count,
                    ),
                )
            ]


        best_labels = None

        best_silhouette_score = -1.0


        for k in candidate_k_values:
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                random_state=42,
            )


            labels = (
                model.fit_predict(
                    feature_matrix
                )
            )


            # ==============================
            # 실루엣 스코어는
            # 2개 이상의 Cluster가
            # 실제로 나뉜 경우에만 계산 가능
            # ==============================

            if (
                len(
                    np.unique(
                        labels
                    )
                )
                < 2
            ):
                continue


            score = silhouette_score(
                feature_matrix,
                labels,
            )


            if (
                score
                > best_silhouette_score
            ):
                best_silhouette_score = (
                    score
                )

                best_labels = labels


        if best_labels is not None:
            return best_labels


        # ==============================
        # 10-4. 모든 후보에서
        # 유효한 Cluster를
        # 만들지 못한 경우 (Fallback)
        # ==============================

        return np.zeros(
            sample_count,
            dtype=int,
        )


    # ==============================
    # 11. Seed 선택
    # ==============================

    @staticmethod
    def _select_seed(
        requests: List[
            MatchingRequest
        ],
    ) -> MatchingRequest:
        """
        조건이 더 제한적인 사용자를
        먼저 그룹화한다.
        """

        return min(
            requests,
            key=lambda request: (
                (
                    request.groupSizeMax
                    - request.groupSizeMin
                ),
                -request.groupSizeMin,
                (
                    request.preferredEndAt
                    - request.preferredStartAt
                ).total_seconds(),
                request.requestId,
            ),
        )


    # ==============================
    # 12. 그룹 크기 후보 결정
    # ==============================

    def _get_target_group_sizes(
        self,
        seed: MatchingRequest,
        remaining: List[
            MatchingRequest
        ],
    ) -> List[
        int
    ]:
        """
        사용자의 최소/최대 그룹 크기와
        남은 인원을 함께 고려한다.

        예:

        6명 모두
        min=3
        max=4

        단순히 4명을 먼저 만들면

        4 + 2

        가 되지만

        3 + 3

        이 가능하므로 3명을 우선한다.
        """

        possible_sizes = list(
            range(
                seed.groupSizeMin,
                seed.groupSizeMax + 1,
            )
        )


        total_remaining = len(
            remaining
        )


        other_requests = [
            request
            for request
            in remaining
            if request.requestId
            != seed.requestId
        ]


        if other_requests:
            minimum_next_group_size = min(
                request.groupSizeMin
                for request
                in other_requests
            )

        else:
            minimum_next_group_size = (
                seed.groupSizeMin
            )


        def size_priority(
            target_size: int,
        ):
            remainder = (
                total_remaining
                - target_size
            )


            # ==============================
            # 딱 맞게 모두 그룹화
            # ==============================

            if remainder == 0:
                priority = 0


            # ==============================
            # 다음 그룹 생성 가능
            # ==============================

            elif (
                remainder
                >= minimum_next_group_size
            ):
                priority = 1


            # ==============================
            # 애매한 잔여 인원 발생
            # ==============================

            else:
                priority = 2


            return (
                priority,
                -target_size,
            )


        return sorted(
            possible_sizes,
            key=size_priority,
        )


    # ==============================
    # 13. 최적 그룹 탐색
    # ==============================

    def _find_best_group(
        self,
        seed: MatchingRequest,
        remaining: List[
            MatchingRequest
        ],
    ) -> Optional[
        List[
            MatchingRequest
        ]
    ]:
        """
        Seed 기준으로

        - 최소/최대 그룹 인원
        - 시간
        - 궁합 점수

        를 고려해 최적 그룹을 찾는다.

        remaining은 이미 같은
        K-Means Cluster로 필터링된
        상태로 전달된다.
        """

        other_requests = [
            request
            for request
            in remaining
            if request.requestId
            != seed.requestId
        ]


        target_sizes = (
            self._get_target_group_sizes(
                seed=seed,
                remaining=remaining,
            )
        )


        # ==============================
        # 그룹 크기별 탐색
        # ==============================

        for target_size in (
            target_sizes
        ):
            eligible_candidates = []


            # ==============================
            # 후보 검사
            # ==============================

            for candidate in (
                other_requests
            ):
                # ==============================
                # 후보자의 그룹 인원 범위 검사
                # ==============================

                if not (
                    candidate.groupSizeMin
                    <= target_size
                    <= candidate.groupSizeMax
                ):
                    continue


                # ==============================
                # 시간이 전혀 겹치지 않으면 제외
                # ==============================

                overlap_seconds = (
                    self
                    ._calculate_datetime_overlap_seconds(
                        seed,
                        candidate,
                    )
                )


                if (
                    overlap_seconds
                    <= 0
                ):
                    continue


                # ==============================
                # 궁합 점수
                # ==============================

                compatibility_score = (
                    self
                    ._calculate_compatibility_score(
                        seed,
                        candidate,
                    )
                )


                eligible_candidates.append(
                    (
                        candidate,
                        compatibility_score,
                    )
                )


            # ==============================
            # 유사도 높은 순
            # ==============================

            eligible_candidates.sort(
                key=lambda item: (
                    item[
                        1
                    ]
                ),
                reverse=True,
            )


            required_count = (
                target_size
                - 1
            )


            if (
                len(
                    eligible_candidates
                )
                < required_count
            ):
                continue


            # ==============================
            # 조합 폭발 방지
            # 상위 12명 내에서 탐색
            # ==============================

            shortlist_limit = min(
                len(
                    eligible_candidates
                ),
                max(
                    required_count,
                    12,
                ),
            )


            shortlist = [
                item[
                    0
                ]
                for item
                in eligible_candidates[
                    :shortlist_limit
                ]
            ]


            best_group = None

            best_score = -1.0


            # ==============================
            # 후보 조합 탐색
            # ==============================

            for candidate_combination in (
                combinations(
                    shortlist,
                    required_count,
                )
            ):
                group = [
                    seed,
                    *candidate_combination,
                ]


                # ==============================
                # 그룹원 전원의
                # 최소/최대 인원 조건 검사
                # ==============================

                if not (
                    self
                    ._is_group_size_valid(
                        group
                    )
                ):
                    continue


                # ==============================
                # 그룹 전체 공통 시간 검사
                # ==============================

                if not (
                    self
                    ._has_common_time_overlap(
                        group
                    )
                ):
                    continue


                # ==============================
                # 그룹 전체 궁합
                # ==============================

                group_score = (
                    self
                    ._calculate_group_compatibility(
                        group
                    )
                )


                if (
                    group_score
                    > best_score
                ):
                    best_score = (
                        group_score
                    )

                    best_group = (
                        group
                    )


            if (
                best_group
                is not None
            ):
                return best_group


        return None


    # ==============================
    # 14. 그룹 인원 조건 검사
    # ==============================

    @staticmethod
    def _is_group_size_valid(
        group: List[
            MatchingRequest
        ],
    ) -> bool:
        """
        현재 그룹의 실제 인원수가

        모든 그룹원의

        groupSizeMin
        ~
        groupSizeMax

        안에 들어가는지 검사한다.
        """

        actual_group_size = len(
            group
        )


        return all(
            member.groupSizeMin
            <= actual_group_size
            <= member.groupSizeMax
            for member
            in group
        )


    # ==============================
    # 15. 그룹 전체 공통 시간 검사
    # ==============================

    @staticmethod
    def _has_common_time_overlap(
        group: List[
            MatchingRequest
        ],
    ) -> bool:
        """
        Pair끼리만 시간이 겹치는 것이 아니라
        그룹원 전체가 동시에 가능한
        시간이 존재하는지 검사한다.
        """

        common_start = max(
            member.preferredStartAt
            for member
            in group
        )


        common_end = min(
            member.preferredEndAt
            for member
            in group
        )


        return (
            common_start
            < common_end
        )


    # ==============================
    # 16. 그룹 전체 궁합
    # ==============================

    def _calculate_group_compatibility(
        self,
        group: List[
            MatchingRequest
        ],
    ) -> float:
        """
        그룹 내 모든 사용자 Pair의
        궁합 점수 평균
        """

        pair_scores = []


        for (
            member_a,
            member_b,
        ) in combinations(
            group,
            2,
        ):
            pair_score = (
                self
                ._calculate_compatibility_score(
                    member_a,
                    member_b,
                )
            )


            pair_scores.append(
                pair_score
            )


        if not pair_scores:
            return 0.0


        return round(
            sum(
                pair_scores
            )
            / len(
                pair_scores
            ),
            2,
        )


    # ==============================
    # 17. 사용자 Pair 궁합
    # ==============================

    def _calculate_compatibility_score(
        self,
        request_a: MatchingRequest,
        request_b: MatchingRequest,
    ) -> float:
        """
        시간       35%
        페이스     30%
        선호 동    20%
        러닝 목표  15%
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
        )


        return round(
            total_score
            * 100,
            2,
        )


    # ==============================
    # 18. 동 Jaccard 유사도
    # ==============================

    @staticmethod
    def _calculate_dong_similarity(
        request_a: MatchingRequest,
        request_b: MatchingRequest,
    ) -> float:
        """
        복수 동 선택에 대해
        Jaccard Similarity 계산

        A = 역삼, 대치
        B = 대치, 삼성

        intersection = 대치
        union = 역삼, 대치, 삼성

        similarity = 1 / 3
        """

        dongs_a = set(
            request_a.regionDongs
        )


        dongs_b = set(
            request_b.regionDongs
        )


        intersection = (
            dongs_a
            & dongs_b
        )


        union = (
            dongs_a
            | dongs_b
        )


        if not union:
            return 0.0


        return (
            len(
                intersection
            )
            / len(
                union
            )
        )


    # ==============================
    # 19. 시간 유사도
    # ==============================

    def _calculate_time_similarity(
        self,
        request_a: MatchingRequest,
        request_b: MatchingRequest,
    ) -> float:
        """
        두 사용자의 러닝 가능 시간 중
        실제 겹치는 비율을 계산한다.
        """

        overlap_seconds = (
            self
            ._calculate_datetime_overlap_seconds(
                request_a,
                request_b,
            )
        )


        if (
            overlap_seconds
            <= 0
        ):
            return 0.0


        duration_a = (
            request_a.preferredEndAt
            - request_a.preferredStartAt
        ).total_seconds()


        duration_b = (
            request_b.preferredEndAt
            - request_b.preferredStartAt
        ).total_seconds()


        shorter_duration = min(
            duration_a,
            duration_b,
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
    # 20. 페이스 유사도
    # ==============================

    @staticmethod
    def _calculate_pace_similarity(
        request_a: MatchingRequest,
        request_b: MatchingRequest,
    ) -> float:
        """
        페이스 중앙값 70%
        페이스 범위 중첩 30%
        """

        center_a = (
            request_a.paceMinSec
            + request_a.paceMaxSec
        ) / 2


        center_b = (
            request_b.paceMinSec
            + request_b.paceMaxSec
        ) / 2


        center_difference = abs(
            center_a
            - center_b
        )


        center_similarity = max(
            0.0,
            1.0
            - (
                center_difference
                / 180
            ),
        )


        # ==============================
        # 범위 교집합
        # ==============================

        overlap_start = max(
            request_a.paceMinSec,
            request_b.paceMinSec,
        )


        overlap_end = min(
            request_a.paceMaxSec,
            request_b.paceMaxSec,
        )


        overlap_length = max(
            0,
            overlap_end
            - overlap_start,
        )


        # ==============================
        # 범위 합집합
        # ==============================

        union_start = min(
            request_a.paceMinSec,
            request_b.paceMinSec,
        )


        union_end = max(
            request_a.paceMaxSec,
            request_b.paceMaxSec,
        )


        union_length = (
            union_end
            - union_start
        )


        if (
            union_length
            == 0
        ):
            range_similarity = (
                1.0
                if center_a
                == center_b
                else 0.0
            )

        else:
            range_similarity = (
                overlap_length
                / union_length
            )


        similarity = (
            center_similarity
            * 0.70
            +
            range_similarity
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
    # 21. 시간 교집합
    # ==============================

    @staticmethod
    def _calculate_datetime_overlap_seconds(
        request_a: MatchingRequest,
        request_b: MatchingRequest,
    ) -> float:
        """
        두 사용자의 공통 가능 시간을
        초 단위로 계산
        """

        overlap_start = max(
            request_a.preferredStartAt,
            request_b.preferredStartAt,
        )


        overlap_end = min(
            request_a.preferredEndAt,
            request_b.preferredEndAt,
        )


        return max(
            0.0,
            (
                overlap_end
                - overlap_start
            ).total_seconds(),
        )


    # ==============================
    # 22. 그룹 공통 동
    # ==============================

    @staticmethod
    def _get_common_region_dongs(
        group: List[
            MatchingRequest
        ],
    ) -> List[
        str
    ]:
        """
        그룹 구성원 모두가 선택한
        공통 동을 반환한다.
        """

        if not group:
            return []


        common_dongs = set(
            group[
                0
            ].regionDongs
        )


        for member in group[
            1:
        ]:
            common_dongs &= set(
                member.regionDongs
            )


        return sorted(
            common_dongs
        )


    # ==============================
    # 23. Response 그룹 생성
    # ==============================

    def _create_group_response(
        self,
        group: List[
            MatchingRequest
        ],
    ) -> Dict:
        """
        완성된 그룹을
        Backend Response 형식으로 변환
        """

        group_score = (
            self
            ._calculate_group_compatibility(
                group
            )
        )


        # ==============================
        # 전체 그룹 공통 시간
        # ==============================

        common_start = max(
            member.preferredStartAt
            for member
            in group
        )


        common_end = min(
            member.preferredEndAt
            for member
            in group
        )


        # ==============================
        # 전체 그룹 공통 동
        # ==============================

        common_region_dongs = (
            self
            ._get_common_region_dongs(
                group
            )
        )


        return {
            "groupNo": 0,

            "memberCount": len(
                group
            ),

            "groupCompatibilityScore": (
                group_score
            ),

            "commonRegionDongs": (
                common_region_dongs
            ),

            "commonStartAt": (
                common_start
            ),

            "commonEndAt": (
                common_end
            ),

            "members": [
                GroupMemberResponse(
                    requestId=(
                        member.requestId
                    ),
                    userId=(
                        member.userId
                    ),
                )
                for member
                in group
            ],
        }