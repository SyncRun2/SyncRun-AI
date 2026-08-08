# ==============================
# 1. 기본 라이브러리
# ==============================

import os
import tempfile

from pathlib import Path


# ==============================
# 2. FastAPI
# ==============================

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)


# ==============================
# 3. OCR 엔진
# ==============================

from running_ocr_engine import (
    RunningOcrEngine,
)


# ==============================
# 4. 매칭 엔진
# ==============================

from matching_kmeans_engine import (
    GoalAwareKMeansGroupMatchingEngine,
)


# ==============================
# 5. Schema
# ==============================

from schemas import (
    MatchBatchResponse,
    MatchingBatchRequest,
    RunningOcrResponse,
)


# ==============================
# 6. FastAPI 앱
# ==============================

app = FastAPI(
    title=(
        "SyncRun AI API"
    ),
    description=(
        "러닝 기록 OCR 및 "
        "K-Means 기반 러닝 그룹 "
        "매칭 AI API"
    ),
    version="2.0.0",
)


# ==============================
# 7. AI 엔진 초기화
# ==============================

ocr_engine = (
    RunningOcrEngine()
)


matching_engine = (
    GoalAwareKMeansGroupMatchingEngine()
)


# ==============================
# 8. 기본 API
# ==============================

@app.get(
    "/"
)
def root():
    """
    서버 기본 정보
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
            "/api/ai/matches/group"
        ),
    }


# ==============================
# 9. Health Check
# ==============================

@app.get(
    "/health"
)
def health_check():
    """
    서버 상태 확인
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
# 10. 러닝 기록 OCR
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
    ),
):
    """
    러닝 기록 이미지에서

    평균 페이스
    총 거리

    를 추출한다.
    """

    # ==============================
    # 10-1. 허용 형식
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
        # 10-2. 이미지 읽기
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


        # ==============================
        # 10-3. 임시 저장
        # ==============================

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
        # 10-4. OCR
        # ==============================

        result = (
            ocr_engine
            .extract_running_data(
                temp_path
            )
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
# 11. AI 그룹 매칭 API
# ==============================

@app.post(
    "/api/ai/matches/group",
    response_model=(
        MatchBatchResponse
    ),
)
def group_running_users(
    request: MatchingBatchRequest,
):
    """
    백엔드에서 현재 MATCHING 상태인
    사용자들을 한 번에 전달받는다.

    AI는 실제로 완성된 그룹만
    Backend에 반환한다.

    그룹에 포함되지 않은 요청은
    Response에 포함하지 않는다.
    """

    result = (
        matching_engine
        .recommend_groups(
            requests=(
                request.requests
            )
        )
    )


    return result