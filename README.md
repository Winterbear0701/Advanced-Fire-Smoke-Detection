# Advanced Fire & Smoke Detection

A Flask-based web application that uses YOLO (Ultralytics) models to detect fire and smoke in images.
Key features:
- Multiple model support (YOLOv8 nano, small, and a custom trained `best.pt`).
- Image and video uploads with annotated output saved in `static/processed`.
- Detection history and basic stats endpoints.
- Simple web UI in `templates/index.html` with drag-and-drop upload, webcam capture, and control panel.

Contents
- `app.py` — main Flask application and detection logic.
- `templates/` — frontend HTML pages (`index.html`).
- `static/` — uploads, processed outputs, and frontend JS/CSS.
- `model/` — stored model weights (e.g. `yolov8n.pt`, `yolov8s.pt`, `best.pt`).
- `requirements.txt` — Python dependencies.

Quickstart (venv, Windows PowerShell)

1. Create and activate a virtual environment (recommended):

	 python -m venv smoke\Scripts\activate

2. Upgrade pip and install dependencies:

	 python -m pip install --upgrade pip; pip install -r requirements.txt

3. (Optional) Place your trained model weights in the `model/` folder. The app will load `model/best.pt`, `model/yolov8s.pt`, and `model/yolov8n.pt` if present. If none are available the code will attempt to download a default YOLOv8 nano model.

4. Run the app:

	 python app.py

5. Open your browser to http://localhost:8080

API Endpoints
- GET /api/models — list available models and whether they're loaded.
- POST /detect — upload an image or video for detection. Form fields:
	- `file` (multipart file) — image (JPG/PNG/GIF) or video (MP4/AVI/MOV)
	- `model_type` (optional) — `yolov8n` (default), `yolov8s`, or `best`
	- `confidence` (optional) — detection confidence threshold (0.1–1.0, default 0.5)
	- `save_results` (optional) — `true`/`false` whether to persist detection history

	Response (JSON): success flag, `uploaded_file`, `processed_file`, detection stats (`detection_count`, `max_confidence`, `fire_count`, `smoke_count`, `risk_level`, etc.)

- GET /api/history — returns recent detection records. Query param: `limit` (default 50).
- GET /api/stats — aggregated statistics computed from saved detection history.

Model files
- The repository includes example weights in `model/`. The app checks for `model/best.pt`, `model/yolov8s.pt`, and `model/yolov8n.pt` on startup.
- To use your own model, drop the `.pt` file into `model/` and name it appropriately (for the UI pick `best` for the custom high-accuracy model).

Docker (optional)
- A `Dockerfile` is included. To build and run the container:

	docker build -t fire-smoke-detect .
	docker run --rm -p 8080:8080 -v %cd%/model:/app/model -v %cd%/static:/app/static fire-smoke-detect

Notes & Troubleshooting
- GPU support: The `requirements.txt` pins CPU PyTorch by default. To enable GPU, install a CUDA-compatible `torch` build matching your CUDA toolkit (for example from https://pytorch.org). Adjust `requirements.txt` accordingly.
- Common errors:
	- "No models available": Ensure model files exist under `model/` or that the app can download `yolov8n.pt` (requires internet).
	- OpenCV issues: On some Windows setups you may need to install `opencv-python-headless` instead of `opencv-python` for headless servers.
- Logs are written to `detection.log` in the project root.

Security
- This project is a demo and not hardened for production. If deploying publicly:
	- Add authentication for uploads and API access.
	- Run behind a reverse proxy and enable TLS.
	- Sanitize or limit uploaded file sizes.

Development notes
- Frontend: `templates/index.html` contains a modern UI with JS that posts to `/detect` and shows results.
- Processing: `process_image_advanced` and `process_video_advanced` annotate media and save results in `static/processed`.

License
- See the `LICENSE` file in the repository root.

Contact
- Maintainer: repository owner

--
Generated/updated README on repo analysis.