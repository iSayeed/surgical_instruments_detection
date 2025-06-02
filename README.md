# Lynqo Surgical Instrument Detection

## Project Overview
Lynqo provides an end-to-end pipeline for surgical instrument detection using computer vision. The project covers data augmentation, model training (YOLO-based), an API for inference, and a user-friendly GUI client. 

---

## Project Structure
- `notebooks/` — Jupyter notebooks for data augmentation and model training
- `datasets/` — Raw and augmented datasets, including Roboflow and custom data
- `src/` — Source code for API, GUI client, and utilities
  - `api.py` — FastAPI server for model inference
  - `gui_client.py` — GUI client for interacting with the API
  - `utils.py` — Utility functions
- `runs/detect/train9/weights/best.pt` — Best trained YOLO model
- `config.json`, `pyproject.toml`, `uv.lock`, `ruff.toml` — Project configuration and dependency management

---

## Setup Instructions

### 1. Install Dependencies
This project uses [uv](https://github.com/astral-sh/uv) for fast Python dependency management. Make sure you have Python 3.10+ installed.

```bash
pip install uv
uv init
uv sync
```

---

## Usage Workflow

### 2. Data Augmentation
- Open `augmentation.ipynb` in the `notebooks/` folder.
- The notebook will guide you to download and augment the dataset. Augmented data will be saved in the appropriate subfolders under `datasets/`.

### 3. Model Training
- Open `model_training.ipynb` in the `notebooks/` folder.
- Train the YOLO model as instructed. You can adjust hyperparameters and experiment with different configurations.
---

## Running the API and GUI Together

### 4. Start the Full Application (API + GUI)
For convenience, you can start both the FastAPI server and the GUI client with a single command using the provided scripts:

#### On macOS/Linux:
```bash
./start_app.sh
```
you might need to grant permission to run 

```bash 
chmod +x start_app.sh
```

#### On Windows:
```bat
start_app.bat
```

- This will launch the API server in the background and then open the GUI client.
- When you close the GUI, the API server will also be stopped automatically.

Alternatively, you can still run each component separately as described below.

---

## Running the API Only

### 5. Start the API Server (Manual)
Run the following command to start the FastAPI server for inference:

```bash
uvicorn src.api:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

- The API supports image upload and returns detection results in JSON format.
- See the `/docs` endpoint for interactive API documentation (Swagger UI).

---

## Using the GUI Client Only

### 6. Launch the GUI Tool (Manual)
A GUI client is provided for convenience, so you do not need to use curl commands in the terminal. 
Make sure the that the API server is up and running.

Start the GUI with:

```bash
uv run src/gui_client.py
```

- The GUI allows you to upload images and view detection results visually.

---

## Additional Notes
- All code is formatted and linted using `ruff`.
- The `config.json` file includes a reference to a surgical tray with instrument name and number, which is assumed be known before an operation. The dataset used for training identifies surgical instruments using integer labels from 0 to 17. A mapping between these numbers and instrument names is provided in the project to help interpret detection results.


