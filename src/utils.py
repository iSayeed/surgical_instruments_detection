from __future__ import annotations

import json
from typing import TYPE_CHECKING
from pathlib import Path

import cv2
from loguru import logger

if TYPE_CHECKING:
    from ultralytics import YOLO


def load_config(config_path: str | Path) -> dict:
    """
    Load configuration from a JSON file.

    Args:
        config_path (str or Path): Path to the config file

    Returns:
        dict: Configuration dictionary

    """
    with open(config_path) as f:
        return json.load(f)
def predict_image(model: YOLO,
                  image_path: str | Path,
                  conf_threshold: float = 0.1,
                  save: bool = True,
                  show: bool = True) -> YOLO:
    """
    Perform prediction on an image using a YOLO model.

    Args:
        model: YOLO model for prediction
        image_path (str): Path to the input image
        conf_threshold (float): Confidence threshold for detections
        save (bool): Whether to save the results
        show (bool): Whether to display the results

    Returns:
        results: YOLO prediction results

    """
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=save,
        save_txt=True,
        save_conf=True,
        show=show,
    )
    return results

def display_detections(results: list, model: YOLO) -> None:
    """
    Display detected objects with their class names and confidence scores.

    Args:
        results: YOLO detection results
        model: YOLO model with class names mapping

    """
    try:
        # Get the names dictionary from the model
        names = model.names

        for r in results:
            logger.info(f"\nDetections in {Path(r.path).name}:")

            if not hasattr(r, "boxes") or len(r.boxes) == 0:
                logger.info("No detections found.")
                continue

            for box in r.boxes:
                # Get the class index and convert to class name
                class_id = int(box.cls[0])
                class_name = names.get(class_id, "Unknown")
                confidence = float(box.conf[0])

                logger.info(f"- {class_name} (Confidence: {confidence:.2f})")

    except Exception as e:
        logger.error(f"Error processing detections: {str(e)}")

def display_surgical_detections(results: list, surgical_instruments: dict) -> dict:
    """
    Display detected surgical instruments with their proper names, counts, and confidence scores.

    Args:
        results: YOLO detection results
        surgical_instruments (dict): Mapping of class indices to instrument names

    Returns:
        dict: Dictionary containing detected instruments and their counts

    """
    output = {"detected_instruments": []}

    try:
        for r in results:
            if not hasattr(r, 'boxes') or len(r.boxes) == 0:  # noqa: Q000
                return {"detected_instruments": []}

            # Dictionary to store counts of each instrument
            instrument_counts = {}

            # Process all detections
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                instrument_name = surgical_instruments.get(str(class_id), f"Unknown Instrument (Class {class_id})")

                # Update counts
                if instrument_name in instrument_counts:
                    instrument_counts[instrument_name]["count"] += 1
                else:
                    instrument_counts[instrument_name] = {
                        "count": 1,
                        "confidence": confidence,
                    }

            # Convert to required format
            for instrument_name, data in instrument_counts.items():
                output["detected_instruments"].append({
                    "type": instrument_name,
                    "count": data["count"],
                })

            # Sort by count (highest first)
            output["detected_instruments"].sort(key=lambda x: x["count"], reverse=True)

            return output

    except Exception as e:
        return {"detected_instruments": [], "error": str(e)}

def visualize_detections(
    image_path: str | Path,
    detections: YOLO.Results,
    surgical_instruments: dict[str, str],
    conf_threshold: float = 0.25,
) -> None:
    """
    Visualize detections on the image with bounding boxes and labels.

    Args:
        image_path (str or Path): Path to the image
        detections: YOLO detection results
        surgical_instruments (dict): Mapping of class indices to instrument names
        conf_threshold (float): Confidence threshold for showing detections

    """
    img = cv2.imread(str(image_path))

    for box in detections.boxes:
        confidence = float(box.conf[0])
        if confidence < conf_threshold:
            continue

        class_id = int(box.cls[0])
        instrument_name = surgical_instruments.get(str(class_id), f"Unknown ({class_id})")

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label
        label = f"{instrument_name} ({confidence:.2%})"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
