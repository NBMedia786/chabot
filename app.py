import os, random, threading, time, mimetypes, base64
from email.message import EmailMessage  # not used for SendGrid; kept for clarity
from pathlib import Path
from datetime import datetime
import requests
from flask import Flask, jsonify, send_from_directory, abort, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
load_dotenv()


ELEVEN_API_KEY   = os.getenv("ELEVEN_API_KEY", "")
AGENT_ID         = os.getenv("AGENT_ID", "")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
FROM_EMAIL       = os.getenv("FROM_EMAIL", "")

if not ELEVEN_API_KEY or not AGENT_ID:
    print("[WARN] ELEVEN_API_KEY / AGENT_ID not set. Live voice will not work until you set them.")


# Paths
ROOT_DIR   = Path(__file__).parent
PUBLIC_DIR = ROOT_DIR / "public"
DATA_DIR   = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_url_path="", static_folder=str(PUBLIC_DIR))

# pip install flask-cors
from flask_cors import CORS
CORS(app, resources={r"/*": {"origins": "http://ronit.nbmedia.co.in/"}})


# ============ Health ============
@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"})


# ============ Frontend ============
@app.get("/")
def index():
    index_path = PUBLIC_DIR / "index.html"
    if not index_path.exists():
        abort(404, description="public/index.html not found")
    return send_from_directory(str(PUBLIC_DIR), "index.html")


@app.get("/<path:path>")
def static_files(path):
    file_path = PUBLIC_DIR / path
    if file_path.exists():
        return send_from_directory(str(PUBLIC_DIR), path)
    abort(404)


# ============ ElevenLabs: WS token ============
@app.get("/conversation-token")
def conversation_token():
    url = "https://api.elevenlabs.io/v1/convai/conversation/token"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Accept": "application/json"}
    params = {"agent_id": AGENT_ID}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        token = data.get("token")
        if not token:
            return jsonify({"error": "Token missing in response", "details": data}), 502
        return jsonify({"token": token})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "ElevenLabs request failed", "details": str(e)}), 502


# ============ Capture + Schedule ============
@app.post("/upload-session")
def upload_session():
    email = (request.form.get("email") or "").strip()
    transcript = (request.form.get("transcript") or "").strip()
    if not email: return jsonify({"error": "Missing email"}), 400
    if not transcript: return jsonify({"error": "Missing transcript"}), 400

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base   = f"session_{stamp}_{secure_filename(email)}"
    (DATA_DIR / f"{base}.txt").write_text(transcript, encoding="utf-8")

    audio_path = None
    if "audio" in request.files and request.files["audio"]:
        f = request.files["audio"]
        ext = Path(f.filename).suffix or (mimetypes.guess_extension(f.mimetype) or ".webm")
        audio_path = DATA_DIR / f"{base}{ext}"
        f.save(audio_path)

    blueprint = _call_gemini_blueprint(transcript)

    delay_seconds = random.randint(10, 30)
    send_at_unix = int(time.time()) + delay_seconds
    _send_email_with_optional_audio(
        to_email=email,
        subject="Your Conversation Blueprint",
        body=blueprint,
        attachment_path=str(audio_path) if audio_path else None,
        send_at_unix=send_at_unix,
    )
    return jsonify({"ok": True, "scheduled_in_seconds": delay_seconds, "send_at_unix": send_at_unix})



# ============ Background Job Helpers ============
def _delayed_blueprint_and_email(email: str, transcript: str, audio_path: str | None, delay_seconds: int):
    try:
        time.sleep(delay_seconds)
        blueprint = _call_gemini_blueprint(transcript)
        _send_email_with_optional_audio(
            to_email=email,
            subject="Your Conversation Blueprint",
            body=blueprint,
            attachment_path=audio_path,
        )
        print(f"[MAIL] Sent to {email}")
    except Exception as e:
        print(f"[ERROR] delayed job failed: {e}")


def _call_gemini_blueprint(transcript: str) -> str:
    """
    Calls Gemini 1.5 Flash to create a concise blueprint from the transcript.
    Falls back to a simple outline if GEMINI_API_KEY is not set or request fails.
    """
    prompt = (
        "You are a conversation analyst. Create a clear, bullet-style blueprint from the user's chat transcript.\n"
        "- Identify goals, blockers, emotions, and decisions.\n"
        "- Extract key actions with owners and deadlines if implied.\n"
        "- Summarize themes in <=10 bullets; include a final 5-step plan.\n\n"
        "USER TRANSCRIPT:\n" + transcript
    )

    if not GEMINI_API_KEY:
        return _fallback_outline(transcript)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return _fallback_outline(transcript)
        text = candidates[0]["content"]["parts"][0]["text"]
        return (text or "").strip() or _fallback_outline(transcript)
    except Exception as e:
        print(f"[WARN] Gemini error: {e}")
        return _fallback_outline(transcript)


def _fallback_outline(transcript: str) -> str:
    lines = [l.strip() for l in (transcript or "").splitlines() if l.strip()]
    head = lines[:10]
    return "Quick Outline:\n- " + "\n- ".join(head or ["(no transcript content)"])


def _send_email_with_optional_audio(to_email, subject, body, attachment_path=None, send_at_unix=None):
    if not SENDGRID_API_KEY: raise RuntimeError("SENDGRID_API_KEY not set")
    if not FROM_EMAIL: raise RuntimeError("FROM_EMAIL not set")

    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": FROM_EMAIL},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }
    if send_at_unix:
        payload["send_at"] = send_at_unix

    if attachment_path and Path(attachment_path).exists():
        mime_type, _ = mimetypes.guess_type(attachment_path)
        mime_type = mime_type or "application/octet-stream"
        with open(attachment_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        payload["attachments"] = [{
            "content": b64, "type": mime_type,
            "filename": Path(attachment_path).name, "disposition": "attachment"
        }]

    r = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type": "application/json"},
        json=payload, timeout=30,
    )
    if r.status_code != 202:
        raise RuntimeError(f"SendGrid error {r.status_code}: {r.text}")
