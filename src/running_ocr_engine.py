# ==============================
# 1. 기본 라이브러리
# ==============================

import os
import re

from typing import (
    List,
    Optional,
)


# ==============================
# 2. OCR 라이브러리
# ==============================

import easyocr


class RunningOcrEngine:
    """
    러닝 기록 이미지에서
    평균 페이스와 총 거리를 추출한다.
    """

    def __init__(
        self,
    ):
        """
        EasyOCR 모델을 초기화한다.

        서버 시작 시 한 번만 로드하여
        반복 요청 처리 속도를 높인다.
        """

        print(
            "EasyOCR 엔진 초기화 중..."
        )

        self.reader = (
            easyocr.Reader(
                [
                    "en",
                    "ko",
                ],
                gpu=False,
            )
        )

        print(
            "EasyOCR 엔진 로드 완료"
        )


    # ==============================
    # 2-1. 러닝 기록 이미지 분석
    # ==============================

    def extract_running_data(
        self,
        image_path: str,
    ) -> dict:
        """
        이미지에서 평균 페이스와
        총 거리를 추출한다.
        """

        if not os.path.exists(
            image_path
        ):
            raise FileNotFoundError(
                "파일을 찾을 수 없습니다: "
                f"{image_path}"
            )

        raw_results: List[str] = (
            self.reader.readtext(
                image_path,
                detail=0,
                paragraph=False,
            )
        )

        print(
            "OCR 원본 텍스트:",
            raw_results,
        )

        average_pace_sec = (
            self._extract_pace_from_results(
                raw_results
            )
        )

        total_distance_km = (
            self._extract_distance_from_results(
                raw_results
            )
        )

        return {
            "average_pace_sec": (
                average_pace_sec
            ),
            "total_distance_km": (
                total_distance_km
            ),
        }


    # ==============================
    # 2-2. 평균 페이스 추출
    # ==============================

    def _extract_pace_from_results(
        self,
        raw_results: List[str],
    ) -> Optional[int]:
        """
        OCR 결과 중 평균 페이스와 가까운
        분:초 형식의 값을 우선 추출한다.
        """

        pace_keywords = [
            "평균 페이스",
            "페이스",
            "average pace",
            "avg pace",
            "pace",
            "/km",
            "min/km",
        ]

        # ==============================
        # 페이스 키워드 주변 텍스트 우선 검사
        # ==============================

        for index, text in enumerate(
            raw_results
        ):
            normalized_text = (
                text.lower()
            )

            has_pace_keyword = any(
                keyword in normalized_text
                for keyword in pace_keywords
            )

            if not has_pace_keyword:
                continue

            window_start = max(
                0,
                index - 1,
            )

            window_end = min(
                len(
                    raw_results
                ),
                index + 3,
            )

            nearby_text = " ".join(
                raw_results[
                    window_start:
                    window_end
                ]
            )

            pace_seconds = (
                self._extract_pace(
                    nearby_text
                )
            )

            if (
                pace_seconds
                is not None
            ):
                return pace_seconds

        # ==============================
        # 전체 OCR 결과 검사
        # ==============================

        combined_text = " ".join(
            raw_results
        )

        return self._extract_pace(
            combined_text
        )


    # ==============================
    # 2-3. 페이스 문자열 파싱
    # ==============================

    @staticmethod
    def _extract_pace(
        text: str,
    ) -> Optional[int]:
        """
        다음 형식의 페이스를 탐색한다.

        5:30
        5：30
        5'30
        5’30”
        5분 30초
        """

        normalized_text = (
            text
            .replace(
                "：",
                ":",
            )
            .replace(
                "’",
                "'",
            )
            .replace(
                "′",
                "'",
            )
            .replace(
                "”",
                '"',
            )
            .replace(
                "″",
                '"',
            )
        )

        pace_patterns = [
            re.compile(
                r"(?<!\d)"
                r"(\d{1,2})"
                r"\s*[:']\s*"
                r"([0-5]\d)"
                r"(?!\d)"
            ),

            re.compile(
                r"(?<!\d)"
                r"(\d{1,2})"
                r"\s*분\s*"
                r"([0-5]?\d)"
                r"\s*초"
            ),
        ]

        for pace_pattern in (
            pace_patterns
        ):
            for match in (
                pace_pattern.finditer(
                    normalized_text
                )
            ):
                minutes = int(
                    match.group(1)
                )

                seconds = int(
                    match.group(2)
                )

                total_seconds = (
                    minutes
                    * 60
                    + seconds
                )

                # 일반적인 러닝 페이스 범위:
                # 2:00/km ~ 15:00/km
                if (
                    120
                    <= total_seconds
                    <= 900
                ):
                    return total_seconds

        return None


    # ==============================
    # 2-4. 총 거리 추출
    # ==============================

    def _extract_distance_from_results(
        self,
        raw_results: List[str],
    ) -> Optional[float]:
        """
        OCR 결과에서 km 단위 거리를 추출한다.
        """

        distance_keywords = [
            "거리",
            "distance",
            "km",
            "킬로미터",
        ]

        # ==============================
        # 거리 키워드 주변 텍스트 우선 검사
        # ==============================

        for index, text in enumerate(
            raw_results
        ):
            normalized_text = (
                text.lower()
            )

            has_distance_keyword = any(
                keyword in normalized_text
                for keyword in distance_keywords
            )

            if not has_distance_keyword:
                continue

            window_start = max(
                0,
                index - 1,
            )

            window_end = min(
                len(
                    raw_results
                ),
                index + 3,
            )

            nearby_text = " ".join(
                raw_results[
                    window_start:
                    window_end
                ]
            )

            distance_km = (
                self._extract_distance(
                    nearby_text
                )
            )

            if (
                distance_km
                is not None
            ):
                return distance_km

        combined_text = " ".join(
            raw_results
        )

        return self._extract_distance(
            combined_text
        )


    # ==============================
    # 2-5. 거리 문자열 파싱
    # ==============================

    @staticmethod
    def _extract_distance(
        text: str,
    ) -> Optional[float]:
        """
        km 또는 킬로미터 형식의
        거리를 추출한다.
        """

        normalized_text = (
            text
            .replace(
                ",",
                ".",
            )
            .replace(
                "㎞",
                "km",
            )
            .replace(
                "ＫＭ",
                "km",
            )
        )

        distance_pattern = (
            re.compile(
                r"(?<!\d)"
                r"(\d{1,3}"
                r"(?:\.\d{1,3})?)"
                r"\s*"
                r"(?:km|킬로미터)"
                r"(?![a-z])",
                re.IGNORECASE,
            )
        )

        for match in (
            distance_pattern.finditer(
                normalized_text
            )
        ):
            distance_km = float(
                match.group(1)
            )

            if (
                0.1
                <= distance_km
                <= 200
            ):
                return round(
                    distance_km,
                    3,
                )

        return None