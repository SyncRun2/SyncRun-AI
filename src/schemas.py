# ==============================
# 1. 기본 라이브러리
# ==============================

from datetime import datetime
from typing import (
    List,
    Optional,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


# ==============================
# 2. OCR Response Schema
# ==============================

class RunningOcrResponse(
    BaseModel
):
    """
    러닝 기록 OCR 결과
    """

    average_pace_sec: Optional[
        int
    ]

    total_distance_km: Optional[
        float
    ]


# ==============================
# 3. 개별 매칭 요청 Schema
# ==============================

class MatchingRequest(
    BaseModel
):
    """
    백엔드에서 AI 서버로 전달하는
    개별 MATCHING 요청 데이터
    """

    requestId: int = Field(
        gt=0,
    )

    userId: int = Field(
        gt=0,
    )

    regionSido: str

    regionSigungu: str

    # ==============================
    # 여러 동 선택 가능
    # ==============================

    regionDongs: List[
        str
    ] = Field(
        min_length=1,
    )

    runningGoal: str

    preferredStartAt: datetime

    preferredEndAt: datetime

    paceMinSec: int = Field(
        ge=60,
        le=1800,
    )

    paceMaxSec: int = Field(
        ge=60,
        le=1800,
    )

    groupSizeMin: int = Field(
        ge=2,
        le=20,
    )

    groupSizeMax: int = Field(
        ge=2,
        le=20,
    )


    # ==============================
    # 3-1. 일반 문자열 검증
    # ==============================

    @field_validator(
        "regionSido",
        "regionSigungu",
        "runningGoal",
    )
    @classmethod
    def validate_string(
        cls,
        value: str,
    ):
        """
        빈 문자열 입력 방지
        """

        value = value.strip()

        if not value:
            raise ValueError(
                "빈 문자열은 "
                "입력할 수 없습니다."
            )

        return value


    # ==============================
    # 3-2. 지역 동 목록 검증
    # ==============================

    @field_validator(
        "regionDongs",
    )
    @classmethod
    def validate_region_dongs(
        cls,
        values: List[
            str
        ],
    ):
        """
        여러 개의 동 선택을 허용한다.

        단,
        같은 동이 중복 입력된 경우에는
        한 번만 유지한다.

        예:
        [
            "역삼동",
            "대치동",
            "역삼동"
        ]

        →

        [
            "역삼동",
            "대치동"
        ]
        """

        cleaned_dongs = []

        seen = set()


        for value in values:
            dong = value.strip()

            if not dong:
                continue

            if dong in seen:
                continue

            seen.add(
                dong
            )

            cleaned_dongs.append(
                dong
            )


        if not cleaned_dongs:
            raise ValueError(
                "regionDongs에는 "
                "최소 1개의 동이 필요합니다."
            )


        return cleaned_dongs


    # ==============================
    # 3-3. 전체 요청 검증
    # ==============================

    @model_validator(
        mode="after"
    )
    def validate_request(
        self,
    ):
        """
        시간 / 페이스 /
        그룹 최소·최대 인원 검증
        """

        # ==============================
        # 시간 범위
        # ==============================

        if (
            self.preferredStartAt
            >= self.preferredEndAt
        ):
            raise ValueError(
                "preferredStartAt은 "
                "preferredEndAt보다 "
                "빨라야 합니다."
            )


        # ==============================
        # 페이스 범위
        # ==============================

        if (
            self.paceMinSec
            > self.paceMaxSec
        ):
            raise ValueError(
                "paceMinSec은 "
                "paceMaxSec보다 "
                "작거나 같아야 합니다."
            )


        # ==============================
        # 그룹 인원 범위
        # ==============================

        if (
            self.groupSizeMin
            > self.groupSizeMax
        ):
            raise ValueError(
                "groupSizeMin은 "
                "groupSizeMax보다 "
                "작거나 같아야 합니다."
            )


        return self


# ==============================
# 4. 전체 MATCHING 요청 Schema
# ==============================

class MatchingBatchRequest(
    BaseModel
):
    """
    백엔드가 현재 MATCHING 상태인
    요청들을 한 번에 AI 서버로 전달한다.
    """

    requests: List[
        MatchingRequest
    ] = Field(
        min_length=1,
    )


    # ==============================
    # 4-1. Request ID 중복 검사
    # ==============================

    @model_validator(
        mode="after"
    )
    def validate_request_ids(
        self,
    ):
        """
        하나의 batch 안에서
        동일한 requestId 중복 방지
        """

        request_ids = [
            request.requestId
            for request
            in self.requests
        ]


        if (
            len(
                request_ids
            )
            != len(
                set(
                    request_ids
                )
            )
        ):
            raise ValueError(
                "동일한 requestId가 "
                "중복되어 있습니다."
            )


        return self


# ==============================
# 5. 그룹 멤버 Response
# ==============================

class GroupMemberResponse(
    BaseModel
):
    """
    AI가 구성한 그룹의 멤버
    """

    requestId: int

    userId: int


# ==============================
# 6. 개별 그룹 Response
# ==============================

class MatchGroupResponse(
    BaseModel
):
    """
    AI가 구성한 하나의 러닝 그룹
    """

    groupNo: int

    memberCount: int

    groupCompatibilityScore: float

    # ==============================
    # 그룹 전체에서 공통으로
    # 선택한 동
    # ==============================

    commonRegionDongs: List[
        str
    ]

    # ==============================
    # 그룹 전체 공통 러닝 가능 시간
    # ==============================

    commonStartAt: datetime

    commonEndAt: datetime

    members: List[
        GroupMemberResponse
    ]


# ==============================
# 7. 전체 매칭 Response
# ==============================

class MatchBatchResponse(
    BaseModel
):
    """
    AI 서버 → Backend

    성공적으로 만들어진 그룹만
    반환한다.

    그룹에 포함되지 않은 사용자는
    반환하지 않는다.
    """

    algorithm: str

    groupCount: int

    groups: List[
        MatchGroupResponse
    ]