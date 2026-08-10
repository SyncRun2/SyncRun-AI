# ==============================
# 1. 기본 라이브러리
# ==============================

import json
import os

from typing import List


# ==============================
# 2. Gemini SDK
# ==============================

from google import genai
from google.genai import types


# ==============================
# 3. Schema
# ==============================

from schemas import (
    CourseRecommendationRequest,
    CourseRecommendationResponse,
    RecommendedCourseResponse,
)


# ==============================
# 4. AI 코스 추천 엔진
# ==============================

RECOMMENDATION_COUNT = 3


class GeminiCourseRecommendationEngine:
    """
    그룹의 공통 지역과 희망 페이스를 바탕으로

    Gemini API를 호출해
    러닝 코스를 추천한다.
    """

    def __init__(self):
        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY가 설정되지 않았습니다. "
                ".env 파일을 확인해주세요."
            )

        self._model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        )

        self._client = genai.Client(
            api_key=api_key,
        )


    # ==============================
    # 4-1. 코스 추천
    # ==============================

    def recommend(
        self,
        request: CourseRecommendationRequest,
    ) -> CourseRecommendationResponse:
        prompt = self._build_prompt(
            request
        )

        response = (
            self._client
            .models
            .generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type=(
                        "application/json"
                    ),
                ),
            )
        )

        courses = self._parse_courses(
            response.text
        )

        return CourseRecommendationResponse(
            courses=courses
        )


    # ==============================
    # 4-2. 프롬프트 생성
    # ==============================

    def _build_prompt(
        self,
        request: CourseRecommendationRequest,
    ) -> str:
        pace_min = self._format_pace(
            request.paceMinSec
        )

        pace_max = self._format_pace(
            request.paceMaxSec
        )

        region_dongs = ", ".join(
            request.regionDongs
        )

        return (
            "당신은 러닝 크루를 위한 코스 추천 전문가입니다.\n"
            f"아래 그룹 정보를 바탕으로 어울리는 러닝 코스를 "
            f"{RECOMMENDATION_COUNT}개 추천해주세요.\n\n"
            f"- 지역: {request.regionSido} "
            f"{request.regionSigungu} "
            f"({region_dongs})\n"
            f"- 그룹 인원: {request.groupSize}명\n"
            f"- 희망 페이스: km당 {pace_min} ~ {pace_max}\n\n"
            "각 코스는 다음 필드를 가진 객체로 구성된 JSON 배열로만 "
            "응답하세요. 다른 설명이나 마크다운은 포함하지 마세요.\n"
            "- name: 코스 이름 (문자열)\n"
            "- regionDescription: 코스가 위치한 지역에 대한 설명 (문자열)\n"
            "- distanceKm: 예상 거리 (숫자, km 단위)\n"
            "- difficulty: 난이도, EASY / MEDIUM / HARD 중 하나 (문자열)\n"
            "- routeDescription: 코스 경로에 대한 간단한 설명 (문자열)\n"
            "- reason: 이 그룹에게 이 코스를 추천하는 이유 (문자열)\n"
        )


    # ==============================
    # 4-3. 페이스 포맷팅
    # ==============================

    def _format_pace(
        self,
        pace_sec: int,
    ) -> str:
        minutes = pace_sec // 60
        seconds = pace_sec % 60

        return f"{minutes}분 {seconds:02d}초"


    # ==============================
    # 4-4. 응답 파싱
    # ==============================

    def _parse_courses(
        self,
        text: str,
    ) -> List[RecommendedCourseResponse]:
        try:
            data = json.loads(
                text
            )
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Gemini 응답을 파싱하지 못했습니다."
            ) from error

        return [
            RecommendedCourseResponse(
                **item
            )
            for item in data
        ]
