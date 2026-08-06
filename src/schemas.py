# ==============================
# 1. 매칭 요청 및 응답 데이터 모델
# ==============================

from datetime import datetime
from typing import List

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


class MatchRequest(
    BaseModel
):
    """
    사용자가 생성하는 러닝 매칭 요청
    """

    regionSido: str = Field(
        min_length=1,
        description="시, 도",
    )

    regionSigungu: str = Field(
        min_length=1,
        description="시, 군, 구",
    )

    regionDongs: List[str] = Field(
        min_length=1,
        description="복수 선택 가능한 선호 읍, 면, 동",
    )

    runningGoal: str = Field(
        min_length=1,
        description="러닝 목표",
    )

    preferredStartAt: datetime = Field(
        description="매칭 시작 시간",
    )

    preferredEndAt: datetime = Field(
        description="매칭 종료 시간",
    )

    paceMinSec: int = Field(
        gt=0,
        description="최소 페이스, sec/km",
    )

    paceMaxSec: int = Field(
        gt=0,
        description="최대 페이스, sec/km",
    )

    groupSize: int = Field(
        ge=2,
        le=20,
        description="요청자를 포함한 희망 인원",
    )

    @field_validator(
        "regionDongs",
    )
    @classmethod
    def validate_region_dongs(
        cls,
        value: List[str],
    ) -> List[str]:
        """
        빈 동 이름을 제거하고,
        같은 동이 반복 입력된 경우 하나로 정리한다.
        """

        cleaned_dongs = []

        for dong in value:
            cleaned_dong = dong.strip()

            if (
                cleaned_dong
                and cleaned_dong
                not in cleaned_dongs
            ):
                cleaned_dongs.append(
                    cleaned_dong
                )

        if len(
            cleaned_dongs
        ) == 0:
            raise ValueError(
                "선호 동을 한 개 이상 선택해야 합니다."
            )

        return cleaned_dongs

    @model_validator(
        mode="after",
    )
    def validate_request_ranges(
        self
    ):
        """
        시간 및 페이스 범위를 검증한다.
        """

        if (
            self.preferredStartAt
            >= self.preferredEndAt
        ):
            raise ValueError(
                "preferredEndAt은 "
                "preferredStartAt보다 늦어야 합니다."
            )

        if (
            self.paceMinSec
            > self.paceMaxSec
        ):
            raise ValueError(
                "paceMinSec은 "
                "paceMaxSec보다 작거나 같아야 합니다."
            )

        return self


class CandidateMatchRequest(
    MatchRequest
):
    """
    DB에서 조회한 매칭 대기 사용자 요청
    """

    requestId: int
    userId: int
    nickname: str


class MatchMemberResponse(
    BaseModel
):
    """
    추천된 사용자 한 명의 결과
    """

    requestId: int
    userId: int
    nickname: str

    regionDongs: List[str]
    commonDongs: List[str]

    runningGoal: str

    paceMinSec: int
    paceMaxSec: int

    preferredStartAt: datetime
    preferredEndAt: datetime

    clusterId: int
    sameCluster: bool

    compatibilityScore: float
    reasons: List[str]


class MatchResponse(
    BaseModel
):
    """
    AI 그룹 매칭 최종 응답
    """

    algorithm: str

    candidateCount: int
    clusterCount: int

    requestedGroupSize: int
    finalGroupSize: int

    groupMembers: List[
        MatchMemberResponse
    ]

    message: str