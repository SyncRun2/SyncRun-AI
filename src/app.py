import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel

from running_ocr_engine import RunningOcrEngine


app = FastAPI()

ocr_engine = RunningOcrEngine()


class RunningOcrResponse(BaseModel):
    average_pace_sec: Optional[int]
    total_distance_km: Optional[float]


@app.post(
    "/ocr/running-record",
    response_model=RunningOcrResponse,
)
async def analyze_running_record(
    file: UploadFile = File(...),
):
    suffix = Path(
        file.filename or "upload.jpg"
    ).suffix

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            image_bytes = await file.read()
            temp_file.write(image_bytes)
            temp_path = temp_file.name

        result = ocr_engine.extract_running_data(
            temp_path
        )

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    finally:
        if (
            temp_path is not None
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)