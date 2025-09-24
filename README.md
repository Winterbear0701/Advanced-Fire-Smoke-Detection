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


Quickstart (Windows PowerShell)

1. Create a virtual environment and activate it (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip; pip install -r requirements.txt
```

3. (Optional) Place your trained model weights in the `model/` folder. The app will load `model/best.pt`, `model/yolov8s.pt`, and `model/yolov8n.pt` if present. If none are available the code may attempt to download a default YOLOv8 nano model (requires internet).

4. Run the app locally (development/simple run):

```powershell
python app.py
```

5. Open your browser to http://localhost:8080

Quick test (CLI)

Use curl (Linux/macOS/Windows with curl) to POST an image to the `/detect` endpoint:

```bash
curl -X POST \
	-F "file=@/full/path/to/image.jpg" \
	-F "model_type=best" \
	-F "confidence=0.5" \
	-F "save_results=true" \
	http://localhost:8080/detect
```

PowerShell example using Invoke-RestMethod:

```powershell
Invoke-RestMethod -Uri http://localhost:8080/detect -Method Post -Form @{
	file = Get-Item 'C:\full\path\to\image.jpg'
	model_type = 'best'
	confidence = '0.5'
	save_results = 'true'
}
```

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

- A `Dockerfile` is included. Build the image locally:

```bash
docker build -t fire-smoke-detect .
```

- Run with mounted model and static folders (Windows PowerShell - replace with your absolute path if you hit issues):

```powershell
# Replace C:\full\path\to\repo with your repository absolute path
docker run --rm -p 8080:8080 -v C:\full\path\to\repo\model:/app/model -v C:\full\path\to\repo\static:/app/static fire-smoke-detect
```

Notes: Docker on Windows sometimes requires using WSL paths or absolute Windows paths for volume mounts; if you use Docker Desktop with WSL2, `-v ${PWD}/model:/app/model` may work from WSL or Git Bash.

Notes & Troubleshooting

- GPU support: `requirements.txt` may reference CPU builds of PyTorch. To enable GPU, install a CUDA-compatible `torch` build matching your CUDA toolkit (see https://pytorch.org) and update your environment accordingly.

- Common errors:
	- "No models available": Ensure model files exist under `model/` (for example `best.pt` or `yolov8n.pt`) or allow the app to download the default model (internet required).
	- OpenCV issues: On some Windows servers you may prefer `opencv-python-headless` for headless deployments.
	- File permissions: Ensure the process can write to `static/uploads`, `static/processed`, and `detection.log`.

- Logs are written to `detection.log` in the project root.

Production note (gunicorn / server)

- `gunicorn` is included in `requirements.txt` but it is not supported on Windows. For production deployments use a Linux host (or WSL) and run with gunicorn or a production ASGI server. Example (Linux):

```bash
gunicorn --workers 4 --bind 0.0.0.0:8080 app:app
```

For async workers or websocket support (if you enable socket features), consider `gunicorn -k gevent` or using `uvicorn`/`hypercorn` depending on your architecture.

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