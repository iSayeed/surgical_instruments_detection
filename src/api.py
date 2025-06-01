from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel
import tempfile
import json
from ultralytics import YOLO

from .utils import predict_image, display_surgical_detections

# Initialize FastAPI app
app = FastAPI(title="Surgical Tools Detection API")

# Load YOLO model
MODEL_PATH = Path("runs/detect/train9/weights/best.pt")
model = YOLO(MODEL_PATH)

# Load reference data and instrument mapping from config.json
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
try:
    with open(CONFIG_PATH) as f:
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
    set_type: str
    actual_weight: float

@app.post("/infer")
async def infer(
    set_type: str = Form(...),
    actual_weight: float = Form(...),
    image: UploadFile = File(...),
):
    try:
        # Validate set type
        if set_type not in REFERENCE_DATA:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid set type. Available types: {list(REFERENCE_DATA.keys())}"
            )
        
        #check if the image is a valid file
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Invalid image format. Only PNG, JPG, and JPEG are allowed.")
        else:
            print(f"Received image: {image.filename}")

        # Save uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_image:
            contents = await image.read()
            temp_image.write(contents)
            temp_image_path = temp_image.name

        # Run inference using utility function
        results = predict_image(
            model=model,
            image_path=temp_image_path,
            conf_threshold=0.1,
            save=False,
            show=False
        )
        
        # Use display_surgical_detections to process results and get detections
        detection_result = display_surgical_detections(results, SURGICAL_INSTRUMENTS)
        
        # Clean up temporary file
        Path(temp_image_path).unlink()

        # Get reference data and expected instruments
        ref_data = REFERENCE_DATA[set_type]
        expected_instruments = ref_data
        
        # Create a map of detected instruments for easy lookup
        detected_map = {item["type"]: item["count"] 
                       for item in detection_result["detected_instruments"]}
        
        # Check for missing or incorrect count items
        missing_items = []
        for expected in expected_instruments:
            expected_type = expected["type"]
            expected_count = expected["expected_count"]
            detected_count = detected_map.get(expected_type, 0)
            
            if detected_count < expected_count:
                missing_items.append({
                    "type": expected_type,
                    "expected": expected_count,
                    "found": detected_count
                })
        
        response = {
            "detected_instruments": detection_result["detected_instruments"],
            "expected_instruments": expected_instruments,
            "set_complete": len(missing_items) == 0,
            "missing_items": missing_items
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    from pathlib import Path
    uvicorn.run(app, host="0.0.0.0", port=8000)
