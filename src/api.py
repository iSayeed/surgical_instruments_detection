from __future__ import annotations

import sys
import json
import shutil
import tempfile
from typing import Annotated
from pathlib import Path
from datetime import datetime
from datetime import timezone

from loguru import logger
from fastapi import File
from fastapi import Form
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import HTTPException
from ultralytics import YOLO
from fastapi.responses import JSONResponse

from .utils import predict_image
from .utils import display_surgical_detections

# Configure logger
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

# Initialize FastAPI app
app = FastAPI(title="Surgical Tools Detection API")

# Setup storage directories
STORAGE_DIR = Path(__file__).parent.parent / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
PREDICTIONS_DIR = STORAGE_DIR / "predictions"
SESSIONS_DIR = STORAGE_DIR / "sessions"

# Create directories
STORAGE_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

# Load reference data and instrument mapping from config.json
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
try:
    with CONFIG_PATH.open() as f:
        config = json.load(f)
        REFERENCE_DATA = config["REFERENCE_DATA"]
        SURGICAL_INSTRUMENTS = config["SURGICAL_INSTRUMENTS"]
        BEST_MODEL = config["BEST_MODEL"]
except FileNotFoundError:
    raise RuntimeError(f"Config file not found at {CONFIG_PATH}")
except json.JSONDecodeError:
    raise RuntimeError("Invalid JSON format in config.json")
except KeyError as e:
    raise RuntimeError(f"Missing required key in config.json: {e}")

# Load YOLO model
MODEL_PATH = Path(f"runs/detect/{BEST_MODEL['folder_name']}/weights/best.pt")
model = YOLO(MODEL_PATH)

def save_session_data(
    original_image: Path,
    predicted_image: Path,
    set_type: str,
    operation_type: str,
    weight_input: float,
    detection_result: dict,
) -> dict:
    """
    Save session data including images and detection results.

    Args:
        original_image: Path to the original uploaded image
        predicted_image: Path to the predicted image
        set_type: Type of surgical set
        operation_type: Type of operation
        weight_input: Weight measurement from input
        detection_result: Detection results from model

    Returns:
        Dictionary with saved file paths and session info

    """
    # Create timestamp-based session directory
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_dir = SESSIONS_DIR / timestamp
    session_dir.mkdir(exist_ok=True)

    # Save original image
    orig_name = f"{timestamp}_original{original_image.suffix}"
    orig_path = UPLOADS_DIR / orig_name
    shutil.copy2(original_image, orig_path)

    # Save predicted image
    pred_name = f"{timestamp}_predicted{predicted_image.suffix}"
    pred_path = PREDICTIONS_DIR / pred_name
    shutil.copy2(predicted_image, pred_path)

    # Prepare session data
    session_data = {
        "timestamp": timestamp,
        "set_type": set_type,
        "operation_type": operation_type,
        "weight_input": weight_input,
        "original_image": str(orig_path),
        "predicted_image": str(pred_path),
        "detection_results": detection_result,
    }

    # Save session data as JSON
    with open(session_dir / "session_data.json", "w") as f:
        json.dump(session_data, f, indent=4)

    return session_data

def check_weight_mismatch(ref_data: list, weight_input: float) -> dict | None:
    """
    Check if the input weight matches the expected weight from reference data.

    Args:
        ref_data: List of expected items including weight
        weight_input: The measured weight from input

    Returns:
        Dict with mismatch details if weight doesn't match, None otherwise

    """
    weight_item = next((item for item in ref_data if "weight" in item), None)
    if not weight_item:
        return None

    expected_weight = float(weight_item["weight"].replace(" kg", ""))
    if weight_input != expected_weight:
        return {
            "type": "Weight",
            "expected": expected_weight,
            "found": weight_input,
        }

    logger.info(f"Weight matches expected: {weight_input} kg")
    return None

@app.post("/infer")
async def infer(
    set_type: Annotated[str, Form()],
    weight_input: Annotated[float, Form()],
    operation_type: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
) -> JSONResponse:
    """
    Process uploaded image for surgical tool detection and validation.

    Args:
        set_type: Type of surgical set to validate against
        weight_input: Weight of the surgical set
        operation_type: Type of operation being performed
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

        # Save session data and get updated paths
        session_data = save_session_data(
            original_image=Path(temp_image_path),
            predicted_image=pred_path,
            set_type=set_type,
            operation_type=operation_type,
            weight_input=weight_input,
            detection_result=detection_result,
        )

        # Update the detection result with the new predicted image path
        detection_result["predicted_image_path"] = session_data["predicted_image"]
        logger.info(f"Found prediction image at: {pred_path}")

        # Clean up temporary file
        Path(temp_image_path).unlink()

        # Get reference data and expected instruments
        ref_data = REFERENCE_DATA[set_type]

        # Filter out the weight entry from expected instruments
        expected_instruments = [item for item in ref_data if "type" in item]

        # Create a map of detected instruments for easy lookup
        detected_map = {
            item["type"]: item["count"]
            for item in detection_result["detected_instruments"]
        }

        # Check for missing or incorrect count items
        missing_items = []

        # Check weight mismatch
        weight_mismatch = check_weight_mismatch(ref_data, weight_input)
        if weight_mismatch:
            missing_items.append(weight_mismatch)

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
            "operation_type": operation_type,
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
