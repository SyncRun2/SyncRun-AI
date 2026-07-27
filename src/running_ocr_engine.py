import os
import re

import easyocr


class RunningOcrEngine:
    """러닝 이미지에서 평균 페이스와 총 거리를 추출한다."""

    def __init__(self):
        print("EasyOCR 엔진 초기화 중...")

        self.reader = easyocr.Reader(
            ["en", "ko"],
            gpu=False,
        )

        print("EasyOCR 엔진 로드 완료")

    def extract_running_data(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"파일을 찾을 수 없습니다: {image_path}"
            )

        raw_results = self.reader.readtext(
            image_path,
            detail=0,
        )

        print(f"OCR 원본 텍스트: {raw_results}")

        combined_text = " ".join(raw_results)

        average_pace_sec = self._extract_pace(
            combined_text
        )

        total_distance_km = self._extract_distance(
            combined_text
        )

        return {
            "average_pace_sec": average_pace_sec,
            "total_distance_km": total_distance_km,
        }

    @staticmethod
    def _extract_pace(text):
        pace_pattern = re.compile(
            r"(\d{1,2})\s*[':]\s*([0-5]\d)"
        )

        for match in pace_pattern.finditer(text):
            minutes = int(match.group(1))
            seconds = int(match.group(2))

            total_seconds = minutes * 60 + seconds

            if 120 <= total_seconds <= 900:
                return total_seconds

        return None

    @staticmethod
    def _extract_distance(text):
        normalized_text = (
            text.replace(",", ".")
            .replace("㎞", "km")
        )

        distance_pattern = re.compile(
            r"(\d{1,3}(?:\.\d{1,3})?)"
            r"\s*(?:km|킬로미터)",
            re.IGNORECASE,
        )

        for match in distance_pattern.finditer(
            normalized_text
        ):
            distance_km = float(match.group(1))

            if 0.1 <= distance_km <= 200:
                return round(distance_km, 3)

        return None

