# ==============================
# 1. 기본 라이브러리
# ==============================

import os
import tempfile

from pathlib import Path
from typing import Optional


# ==============================
# 2. FastAPI 및 Pydantic
# ==============================

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from pydantic import BaseModel


# ==============================
# 3. AI 엔진 및 데이터 모듈
# ==============================

from matching_kmeans_engine import (
    GoalAwareKMeansMatchingEngine,
)

from mock_repository import (
    MockMatchRepository,
)

from running_ocr_engine import (
    RunningOcrEngine,
)

from schemas import (
    MatchRequest,
    MatchResponse,
)


# ==============================
# 4. FastAPI 앱 생성
# ==============================

app = FastAPI(
    title=(
        "SyncRun AI API"
    ),
    description=(
        "러닝 기록 OCR 및 "
        "K-Means 기반 러닝 그룹 추천 API"
    ),
    version="1.0.0",
)


# ==============================
# 5. AI 엔진 및 저장소 생성
# ==============================

ocr_engine = (
    RunningOcrEngine()
)

matching_engine = (
    GoalAwareKMeansMatchingEngine()
)

match_repository = (
    MockMatchRepository()
)


# ==============================
# 6. OCR 응답 데이터 모델
# ==============================

class RunningOcrResponse(
    BaseModel
):
    """
    러닝 기록 OCR 분석 결과
    """

    average_pace_sec: Optional[
        int
    ]

    total_distance_km: Optional[
        float
    ]


# ==============================
# 7. 기본 접속 API
# ==============================

@app.get(
    "/"
)
def root():
    """
    기본 주소 접속 결과를 반환한다.
    """

    return {
        "status": "ok",
        "message": (
            "SyncRun AI API가 "
            "실행 중입니다."
        ),
        "swagger": "/docs",
        "health": "/health",
        "ocr": (
            "/ocr/running-record"
        ),
        "matching": (
            "/api/ai/matches/recommend"
        ),
    }


# ==============================
# 8. 서버 상태 확인 API
# ==============================

@app.get(
    "/health"
)
def health_check():
    """
    서버 실행 상태를 확인한다.
    """

    return {
        "status": "ok",
        "service": (
            "SyncRun AI API"
        ),
        "ocrEngine": "ready",
        "matchingEngine": "ready",
    }


# ==============================
# 9. 러닝 기록 OCR API
# ==============================

@app.post(
    "/ocr/running-record",
    response_model=(
        RunningOcrResponse
    ),
)
async def analyze_running_record(
    file: UploadFile = File(
        ...,
        description=(
            "러닝 기록 스크린샷"
        ),
    ),
):
    """
    러닝 기록 이미지를 분석하여
    평균 페이스와 총 거리를 반환한다.
    """

    # ==============================
    # 9-1. 이미지 형식 검사
    # ==============================

    allowed_content_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    content_type = (
        file.content_type
        or ""
    ).lower()

    if (
        content_type
        not in allowed_content_types
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "JPG, JPEG, PNG, WEBP "
                "이미지만 업로드할 수 있습니다."
            ),
        )

    original_filename = (
        file.filename
        or "upload.jpg"
    )

    suffix = (
        Path(
            original_filename
        )
        .suffix
        .lower()
    )

    allowed_suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    if (
        suffix
        not in allowed_suffixes
    ):
        suffix = ".jpg"

    temp_path = None

    try:
        # ==============================
        # 9-2. 이미지 임시 저장
        # ==============================

        image_bytes = (
            await file.read()
        )

        if len(
            image_bytes
        ) == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "업로드된 이미지가 "
                    "비어 있습니다."
                ),
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(
                image_bytes
            )

            temp_path = (
                temp_file.name
            )

        # ==============================
        # 9-3. OCR 분석
        # ==============================

        result = (
            ocr_engine
            .extract_running_data(
                image_path=(
                    temp_path
                )
            )
        )

        # ==============================
        # 9-4. 추출 결과 확인
        # ==============================

        if (
            result[
                "average_pace_sec"
            ]
            is None
            and result[
                "total_distance_km"
            ]
            is None
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "이미지에서 평균 페이스와 "
                    "총 거리를 찾지 못했습니다."
                ),
            )

        return result

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(
                error
            ),
        ) from error

    finally:
        # ==============================
        # 9-5. 임시 파일 정리
        # ==============================

        await file.close()

        if (
            temp_path is not None
            and os.path.exists(
                temp_path
            )
        ):
            os.remove(
                temp_path
            )


# ==============================
# 10. AI 러닝 그룹 추천 API
# ==============================

@app.post(
    "/api/ai/matches/recommend",
    response_model=MatchResponse,
)
def recommend_running_group(
    request: MatchRequest,
):
    """
    현재 사용자의 매칭 요청을 기반으로
    AI 러닝 그룹을 추천한다.
    """

    pending_requests = (
        match_repository
        .get_pending_requests()
    )

    result = (
        matching_engine
        .recommend_group(
            target_request=(
                request
            ),
            pending_requests=(
                pending_requests
            ),
        )
    )

    return result