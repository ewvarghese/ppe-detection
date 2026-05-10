import json
import os
import re
import smtplib
import time
from datetime import datetime
from email.message import EmailMessage
from threading import Event, Lock, Thread
from uuid import uuid4

import cv2
import torch
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, url_for
from playsound3 import playsound
from ultralytics import YOLO

# -------------------------
# CONFIGURATION
# -------------------------
MODEL_PATH = "runs/detect/ppe_train_10_epochs_final/weights/best.pt"
VIOLATION_CLASSES = ["no-helmet", "no-gloves", "no-vest", "no-boots", "no-goggles"]
EMAIL_SENDER = "tl22bime0393@vidyaacademy.ac.in"
EMAIL_PASSWORD = "6238535521"
EMAIL_RECEIVER = "aslam12r3@gmail.com"
CAMERA_CONFIG_PATH = "camera_sources.json"
DEFAULT_CAMERA_URL = "http://10.205.245.204:5000/video_feed"
DETECTION_CONFIDENCE = 0.25
ALERT_CONFIDENCE = 0.5
CAMERA_RETRY_SECONDS = 5
NO_CAMERA_WAIT_SECONDS = 2
FRAME_POLL_SECONDS = 0.1
ALERT_COOLDOWN_SECONDS = 15

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "ppe-monitor-secret-key")

violation_folder = "static/violations"
os.makedirs(violation_folder, exist_ok=True)

camera_lock = Lock()
state_lock = Lock()

camera_frames = {}
camera_status = {}
alert_timestamps = {}
detection_workers = {}


def default_camera_config():
    default_camera = {
        "id": "default-camera",
        "name": "Main Site Camera",
        "url": DEFAULT_CAMERA_URL,
    }
    return {"active_camera_id": default_camera["id"], "cameras": [default_camera]}


def ensure_camera_config():
    if os.path.exists(CAMERA_CONFIG_PATH):
        return

    save_camera_config(default_camera_config())


def load_camera_config():
    ensure_camera_config()
    with open(CAMERA_CONFIG_PATH, "r", encoding="utf-8") as config_file:
        data = json.load(config_file)

    data.setdefault("cameras", [])
    data.setdefault("active_camera_id", None)

    if data["cameras"] and not data["active_camera_id"]:
        data["active_camera_id"] = data["cameras"][0]["id"]

    return data


def save_camera_config(config):
    with open(CAMERA_CONFIG_PATH, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)


def get_camera_state():
    with camera_lock:
        return load_camera_config()


def get_camera(camera_id):
    config = get_camera_state()
    for camera in config.get("cameras", []):
        if camera["id"] == camera_id:
            return camera
    return None


def get_active_camera():
    config = get_camera_state()
    active_id = config.get("active_camera_id")
    for camera in config.get("cameras", []):
        if camera["id"] == active_id:
            return camera
    return None


def add_camera(name, url):
    with camera_lock:
        config = load_camera_config()
        new_camera = {"id": uuid4().hex, "name": name.strip(), "url": url.strip()}
        config["cameras"].append(new_camera)
        if not config.get("active_camera_id"):
            config["active_camera_id"] = new_camera["id"]
        save_camera_config(config)
    return new_camera


def set_active_camera(camera_id):
    with camera_lock:
        config = load_camera_config()
        if any(camera["id"] == camera_id for camera in config.get("cameras", [])):
            config["active_camera_id"] = camera_id
            save_camera_config(config)
            return True
    return False


def remove_camera(camera_id):
    with camera_lock:
        config = load_camera_config()
        cameras = config.get("cameras", [])
        filtered_cameras = [camera for camera in cameras if camera["id"] != camera_id]

        if len(filtered_cameras) == len(cameras):
            return False

        config["cameras"] = filtered_cameras
        if config.get("active_camera_id") == camera_id:
            config["active_camera_id"] = filtered_cameras[0]["id"] if filtered_cameras else None

        save_camera_config(config)
        return True


def build_dashboard_stats(images):
    stats = {label: 0 for label in VIOLATION_CLASSES}
    for image_name in images:
        for label in VIOLATION_CLASSES:
            if label in image_name:
                stats[label] += 1
    return stats


def build_dashboard_payload():
    images = sorted(os.listdir(violation_folder), reverse=True)
    stats = build_dashboard_stats(images)
    config = get_camera_state()
    active_camera = get_active_camera()
    cameras = config.get("cameras", [])

    with state_lock:
        statuses = {camera["id"]: camera_status.get(camera["id"], "starting") for camera in cameras}

    return {
        "images": images,
        "stats": stats,
        "total": len(images),
        "last": format_detection_time(images),
        "active_camera": active_camera,
        "camera_count": len(cameras),
        "cameras": cameras,
        "camera_statuses": statuses,
    }


def format_detection_time(images):
    if not images:
        return "No detections yet"

    match = re.search(r"(\d{8}_\d{6})", images[0])
    if not match:
        return images[0]

    try:
        detected_at = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        return detected_at.strftime("%d %b %Y, %I:%M:%S %p")
    except ValueError:
        return images[0]


def slugify_camera_name(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "camera"


def set_camera_status(camera_id, status):
    with state_lock:
        camera_status[camera_id] = status


def set_camera_frame(camera_id, frame):
    with state_lock:
        camera_frames[camera_id] = None if frame is None else frame.copy()


def get_camera_frame(camera_id):
    with state_lock:
        frame = camera_frames.get(camera_id)
        return None if frame is None else frame.copy()


def should_send_alert(camera_id, class_name):
    now = time.time()
    key = (camera_id, class_name)

    with state_lock:
        last_sent = alert_timestamps.get(key, 0)
        if now - last_sent < ALERT_COOLDOWN_SECONDS:
            return False

        alert_timestamps[key] = now
        return True


# -------------------------
# EMAIL & DETECTION
# -------------------------
def send_email_alert(image_path):
    try:
        msg = EmailMessage()
        msg["Subject"] = "PPE Violation Detected"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg.set_content("A PPE violation has been detected.")
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        msg.add_attachment(
            image_data,
            maintype="image",
            subtype="jpeg",
            filename="violation.jpg",
        )
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
    except Exception as exc:
        print(f"Email failed: {exc}")


def detect_camera_stream(camera, stop_event):
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    camera_id = camera["id"]
    camera_name = camera["name"]
    video_source = camera["url"]
    camera_slug = slugify_camera_name(camera_name)

    while not stop_event.is_set():
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            set_camera_status(camera_id, "offline")
            set_camera_frame(camera_id, None)
            print(f"Could not connect to {camera_name} ({video_source}). Retrying in {CAMERA_RETRY_SECONDS}s...")
            stop_event.wait(CAMERA_RETRY_SECONDS)
            continue

        set_camera_status(camera_id, "live")
        print(f"Connected to {camera_name} ({video_source})")

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                set_camera_status(camera_id, "reconnecting")
                set_camera_frame(camera_id, None)
                print(f"Stream lost for {camera_name}. Attempting to reconnect...")
                break

            results = model(frame, device=device, imgsz=640, conf=DETECTION_CONFIDENCE, verbose=False)
            annotated_frame = results[0].plot()

            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                if class_name in VIOLATION_CLASSES and confidence >= ALERT_CONFIDENCE and should_send_alert(camera_id, class_name):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{camera_slug}_{class_name}_{timestamp}.jpg"
                    filepath = os.path.join(violation_folder, filename)
                    cv2.imwrite(filepath, frame)

                    Thread(target=playsound, args=("alert.wav",), daemon=True).start()
                    Thread(target=send_email_alert, args=(filepath,), daemon=True).start()

            set_camera_frame(camera_id, annotated_frame)

        cap.release()

    set_camera_status(camera_id, "stopped")
    set_camera_frame(camera_id, None)


def ensure_detection_workers():
    while True:
        config = get_camera_state()
        configured_cameras = {camera["id"]: camera for camera in config.get("cameras", [])}

        with state_lock:
            running_camera_ids = set(detection_workers.keys())

        for camera_id, camera in configured_cameras.items():
            if camera_id in running_camera_ids:
                continue

            stop_event = Event()
            worker = Thread(target=detect_camera_stream, args=(camera, stop_event), daemon=True)

            with state_lock:
                detection_workers[camera_id] = {"thread": worker, "stop_event": stop_event}
                camera_status[camera_id] = "starting"
                camera_frames.setdefault(camera_id, None)

            worker.start()

        removed_camera_ids = running_camera_ids - set(configured_cameras.keys())
        for camera_id in removed_camera_ids:
            with state_lock:
                worker_info = detection_workers.pop(camera_id, None)
                camera_status.pop(camera_id, None)
                camera_frames.pop(camera_id, None)
                keys_to_remove = [key for key in alert_timestamps if key[0] == camera_id]
                for key in keys_to_remove:
                    alert_timestamps.pop(key, None)

            if worker_info:
                worker_info["stop_event"].set()

        if not configured_cameras:
            time.sleep(NO_CAMERA_WAIT_SECONDS)
        else:
            time.sleep(1)


# -------------------------
# FLASK ROUTES
# -------------------------
def generate_frames(camera_id):
    while True:
        frame = get_camera_frame(camera_id)

        if frame is None:
            time.sleep(FRAME_POLL_SECONDS)
            continue

        flag, encoded_image = cv2.imencode(".jpg", frame)
        if not flag:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encoded_image) + b"\r\n"
        )


@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    if not get_camera(camera_id):
        return ("Camera not found", 404)
    return Response(generate_frames(camera_id), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def dashboard():
    return render_template("dashboard_pro.html", **build_dashboard_payload())


@app.get("/api/dashboard-state")
def dashboard_state():
    payload = build_dashboard_payload()
    payload["image_urls"] = [
        url_for("static", filename=f"violations/{image_name}")
        for image_name in payload["images"][:24]
    ]
    payload["video_urls"] = {
        camera["id"]: url_for("video_feed", camera_id=camera["id"])
        for camera in payload["cameras"]
    }
    return jsonify(payload)


@app.route("/cameras", methods=["GET", "POST"])
def camera_management():
    if request.method == "POST":
        camera_name = request.form.get("camera_name", "").strip()
        camera_url = request.form.get("camera_url", "").strip()

        if not camera_name or not camera_url:
            flash("Camera name and IP stream URL are required.", "danger")
        else:
            add_camera(camera_name, camera_url)
            flash("Camera added successfully. Detection will start automatically for this camera.", "success")

        return redirect(url_for("camera_management"))

    config = get_camera_state()
    with state_lock:
        statuses = {
            camera["id"]: camera_status.get(camera["id"], "starting")
            for camera in config.get("cameras", [])
        }

    return render_template(
        "camera_management.html",
        cameras=config.get("cameras", []),
        active_camera_id=config.get("active_camera_id"),
        camera_statuses=statuses,
    )


@app.get("/api/cameras")
def camera_list():
    config = get_camera_state()
    cameras = config.get("cameras", [])
    with state_lock:
        statuses = {
            camera["id"]: camera_status.get(camera["id"], "starting")
            for camera in cameras
        }

    return jsonify(
        {
            "active_camera_id": config.get("active_camera_id"),
            "cameras": cameras,
            "camera_statuses": statuses,
        }
    )


@app.post("/cameras/<camera_id>/activate")
def activate_camera(camera_id):
    if set_active_camera(camera_id):
        flash("Featured camera updated.", "success")
    else:
        flash("Selected camera could not be found.", "danger")
    return redirect(url_for("camera_management"))


@app.post("/cameras/<camera_id>/delete")
def delete_camera(camera_id):
    if remove_camera(camera_id):
        flash("Camera removed.", "success")
    else:
        flash("Camera could not be removed.", "danger")
    return redirect(url_for("camera_management"))


if __name__ == "__main__":
    ensure_camera_config()
    detection_manager_thread = Thread(target=ensure_detection_workers, daemon=True)
    detection_manager_thread.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
