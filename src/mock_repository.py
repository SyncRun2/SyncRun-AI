# ==============================
# 1. 테스트용 매칭 대기 사용자 저장소
# ==============================

from typing import List

from schemas import (
    CandidateMatchRequest,
)


class MockMatchRepository:
    """
    실제 DB 연동 전 사용하는
    테스트용 매칭 대기 요청 저장소
    """

    def get_pending_requests(
        self,
    ) -> List[
        CandidateMatchRequest
    ]:
        """
        매칭 대기 상태인 사용자 요청 목록을 반환한다.
        """

        raw_requests = [
            # ==============================
            # 강남구 / 시간 일치 후보
            # ==============================
            {
                "requestId": 101,
                "userId": 1001,
                "nickname": "러너민수",

                "regionSido": "서울특별시",
                "regionSigungu": "강남구",

                "regionDongs": [
                    "역삼동",
                    "대치동",
                ],

                "runningGoal": (
                    "RACE_PREPARATION"
                ),

                "preferredStartAt": (
                    "2026-08-01T18:10:00"
                ),

                "preferredEndAt": (
                    "2026-08-01T19:50:00"
                ),

                "paceMinSec": 335,
                "paceMaxSec": 365,

                "groupSize": 4,
            },
            {
                "requestId": 102,
                "userId": 1002,
                "nickname": "달리는지수",

                "regionSido": "서울특별시",
                "regionSigungu": "강남구",

                "regionDongs": [
                    "대치동",
                    "삼성동",
                ],

                "runningGoal": (
                    "RACE_PREPARATION"
                ),

                "preferredStartAt": (
                    "2026-08-01T18:30:00"
                ),

                "preferredEndAt": (
                    "2026-08-01T20:00:00"
                ),

                "paceMinSec": 340,
                "paceMaxSec": 370,

                "groupSize": 4,
            },
            {
                "requestId": 103,
                "userId": 1003,
                "nickname": "건강러너",

                "regionSido": "서울특별시",
                "regionSigungu": "강남구",

                "regionDongs": [
                    "삼성동",
                    "청담동",
                ],

                "runningGoal": "GENERAL",

                "preferredStartAt": (
                    "2026-08-01T18:00:00"
                ),

                "preferredEndAt": (
                    "2026-08-01T19:00:00"
                ),

                "paceMinSec": 390,
                "paceMaxSec": 420,

                "groupSize": 4,
            },
            {
                "requestId": 104,
                "userId": 1004,
                "nickname": "페이스메이커",

                "regionSido": "서울특별시",
                "regionSigungu": "강남구",

                "regionDongs": [
                    "역삼동",
                    "논현동",
                ],

                "runningGoal": (
                    "RACE_PREPARATION"
                ),

                "preferredStartAt": (
                    "2026-08-01T17:50:00"
                ),

                "preferredEndAt": (
                    "2026-08-01T19:30:00"
                ),

                "paceMinSec": 325,
                "paceMaxSec": 355,

                "groupSize": 4,
            },
            {
                "requestId": 105,
                "userId": 1005,
                "nickname": "퇴근런",

                "regionSido": "서울특별시",
                "regionSigungu": "강남구",

                "regionDongs": [
                    "논현동",
                    "신사동",
                ],

                "runningGoal": "GENERAL",

                "preferredStartAt": (
                    "2026-08-01T19:00:00"
                ),

                "preferredEndAt": (
                    "2026-08-01T21:00:00"
                ),

                "paceMinSec": 350,
                "paceMaxSec": 380,

                "groupSize": 4,
            },
            {
                "requestId": 106,
                "userId": 1006,
                "nickname": "역삼러너",

                "regionSido": "서울특별시",
                "regionSigungu": "강남구",

                "regionDongs": [
                    "역삼동",
                ],

                "runningGoal": "GENERAL",

                "preferredStartAt": (
                    "2026-08-01T18:00:00"
                ),

                "preferredEndAt": (
                    "2026-08-01T20:00:00"
                ),

                "paceMinSec": 330,
                "paceMaxSec": 360,

                "groupSize": 4,
            },
            {
                "requestId": 107,
                "userId": 1007,
                "nickname": "저녁레이서",

                "regionSido": "서울특별시",
                "regionSigungu": "강남구",

                "regionDongs": [
                    "역삼동",
                    "삼성동",
                ],

                "runningGoal": (
                    "RACE_PREPARATION"
                ),

                "preferredStartAt": (
                    "2026-08-01T18:20:00"
                ),

                "preferredEndAt": (
                    "2026-08-01T20:10:00"
                ),

                "paceMinSec": 332,
                "paceMaxSec": 358,

                "groupSize": 4,
            },

            # ==============================
            # 다른 구이므로 제외되는 사용자
            # ==============================
            {
                "requestId": 108,
                "userId": 1008,
                "nickname": "서초러너",

                "regionSido": "서울특별시",
                "regionSigungu": "서초구",

                "regionDongs": [
                    "서초동",
                ],

                "runningGoal": (
                    "RACE_PREPARATION"
                ),

                "preferredStartAt": (
                    "2026-08-01T18:00:00"
                ),

                "preferredEndAt": (
                    "2026-08-01T20:00:00"
                ),

                "paceMinSec": 330,
                "paceMaxSec": 360,

                "groupSize": 4,
            },

            # ==============================
            # 날짜가 달라 제외되는 사용자
            # ==============================
            {
                "requestId": 109,
                "userId": 1009,
                "nickname": "다음날러너",

                "regionSido": "서울특별시",
                "regionSigungu": "강남구",

                "regionDongs": [
                    "역삼동",
                ],

                "runningGoal": (
                    "RACE_PREPARATION"
                ),

                "preferredStartAt": (
                    "2026-08-02T18:00:00"
                ),

                "preferredEndAt": (
                    "2026-08-02T20:00:00"
                ),

                "paceMinSec": 330,
                "paceMaxSec": 360,

                "groupSize": 4,
            },

            # ==============================
            # 시간이 겹치지 않아 제외되는 사용자
            # ==============================
            {
                "requestId": 110,
                "userId": 1010,
                "nickname": "아침러너",

                "regionSido": "서울특별시",
                "regionSigungu": "강남구",

                "regionDongs": [
                    "역삼동",
                    "대치동",
                ],

                "runningGoal": (
                    "RACE_PREPARATION"
                ),

                "preferredStartAt": (
                    "2026-08-01T07:00:00"
                ),

                "preferredEndAt": (
                    "2026-08-01T09:00:00"
                ),

                "paceMinSec": 330,
                "paceMaxSec": 360,

                "groupSize": 4,
            },
        ]

        pending_requests = [
            CandidateMatchRequest(
                **request
            )
            for request in raw_requests
        ]

        return pending_requests