from __future__ import annotations

import sys
import json
import tempfile
from typing import Annotated
from pathlib import Path

from loguru import logger
from fastapi import File
from fastapi import Form
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from fastapi.responses import JSONResponse

from .utils import predict_image
from .utils import display_surgical_detections

# Configure logger
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

# Initialize FastAPI app
app = FastAPI(title="Surgical Tools Detection API")

# Load YOLO model
MODEL_PATH = Path("runs/detect/train9/weights/best.pt")
model = YOLO(MODEL_PATH)

# Load reference data and instrument mapping from config.json
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
try:
    with CONFIG_PATH.open() as f:
        config = json.load(f)
        REFERENCE_DATA = config["REFERENCE_DATA"]
        SURGICAL_INSTRUMENTS = config["SURGICAL_INSTRUMENTS"]
except FileNotFoundError:
    raise RuntimeError(f"Config file not found at {CONFIG_PATH}")
except json.JSONDecodeError:
    raise RuntimeError("Invalid JSON format in config.json")
except KeyError as e:
    raise RuntimeError(f"Missing required key in config.json: {e}")


class InferenceRequest(BaseModel):
    """Request model for surgical tool inference."""

    set_type: str
    actual_weight: float  # Required for future weight validation


@app.post("/infer")
async def infer(
    set_type: Annotated[str, Form()],
    actual_weight: Annotated[float, Form()],  # Required for future weight validation
    image: Annotated[UploadFile, File()],
) -> JSONResponse:
    """
    Process uploaded image for surgical tool detection and validation.

    Args:
        set_type: Type of surgical set to validate against
        actual_weight: Weight of the surgical set
        image: Uploaded image file for detection

    Returns:
        JSONResponse containing detection results and validation status

    """
    try:
        # Validate set type
        if set_type not in REFERENCE_DATA:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid set type. Available types: {list(REFERENCE_DATA.keys())}",
            )

        # check if the image is a valid file
        if not image.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Only PNG, JPG, and JPEG are allowed.",
            )

        logger.info(f"Received image: {image.filename}")

        # Save uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
            contents = await image.read()
            temp_image.write(contents)
            temp_image_path = temp_image.name

        # Run inference using utility function
        results = predict_image(
            model=model,
            image_path=temp_image_path,
            conf_threshold=0.1,
            save=True,
            show=False,
        )

        # Get the absolute path of the project root directory
        project_root = Path(__file__).parent.parent.absolute()
        detect_dir = project_root / "runs/detect"

        # Find the latest prediction folder (predict, predict2, predict3, etc.)
        pred_folders = sorted(
            [d for d in detect_dir.glob("predict*") if d.is_dir()],
            key=lambda x: int(x.name.replace("predict", "") or "0"),
        )

        if not pred_folders:
            raise HTTPException(status_code=500, detail="No prediction folders found")

        latest_pred_folder = pred_folders[-1]
        logger.info(f"Looking for predictions in: {latest_pred_folder}")

        # Get the prediction image from the latest folder
        pred_path = next(latest_pred_folder.glob("*.jpg"), None)
        if not pred_path:
            raise HTTPException(
                status_code=500,
                detail=f"No prediction image found in {latest_pred_folder}",
            )

        # Use display_surgical_detections to process results and get detections
        detection_result = display_surgical_detections(results, SURGICAL_INSTRUMENTS)

        # Add the absolute predicted image path to the detection result
        detection_result["predicted_image_path"] = str(pred_path)
        logger.info(f"Found prediction image at: {pred_path}")

        # Clean up temporary file
        Path(temp_image_path).unlink()

        # Get reference data and expected instruments
        ref_data = REFERENCE_DATA[set_type]
        expected_instruments = ref_data

        # Create a map of detected instruments for easy lookup
        detected_map = {
            item["type"]: item["count"]
            for item in detection_result["detected_instruments"]
        }

        # Check for missing or incorrect count items
        missing_items = []
        for expected in expected_instruments:
            expected_type = expected["type"]
            expected_count = expected["expected_count"]
            detected_count = detected_map.get(expected_type, 0)

            if detected_count < expected_count:
                missing_items.append(
                    {
                        "type": expected_type,
                        "expected": expected_count,
                        "found": detected_count,
                    },
                )

        response = {
            "detected_instruments": detection_result["detected_instruments"],
            "expected_instruments": expected_instruments,
            "set_complete": len(missing_items) == 0,
            "missing_items": missing_items,
            "predicted_image_path": detection_result["predicted_image_path"],
        }

        logger.info(f"Sending response with image path: {response['predicted_image_path']}")
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except (OSError, ValueError) as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.warning("Running server with localhost")
    uvicorn.run(app, host="127.0.0.1", port=8000)
