

import os
import random
import threading
import time
import uuid
import smtplib
import ssl
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict

from flask import Flask, jsonify, send_from_directory, abort, request, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter, Retry

# ── Supabase client
from supabase import create_client, Client  # type: ignore

# =========================
# Environment / Config
# =========================
load_dotenv()

# Core config
ELEVEN_API_KEY = (os.getenv("ELEVEN_API_KEY") or "").strip()
AGENT_ID       = (os.getenv("AGENT_ID") or os.getenv("ELEVEN_AGENT_ID") or "").strip()
PORT           = int(os.getenv("PORT", "5000"))

# Email config
SENDGRID_API_KEY = (os.getenv("SENDGRID_API_KEY") or "").strip()
FROM_EMAIL       = (os.getenv("FROM_EMAIL") or "info@example.com").strip()
FROM_NAME        = (os.getenv("FROM_NAME") or "AI Voice Coach").strip()
SENDGRID_SANDBOX = (os.getenv("SENDGRID_SANDBOX") or "false").lower() == "true"

# SMTP fallback (optional)
SMTP_HOST     = os.getenv("SMTP_HOST") or ""
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER") or ""
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD") or ""
SMTP_TLS      = (os.getenv("SMTP_TLS") or "true").lower() == "true"
REPLY_TO      = os.getenv("REPLY_TO") or ""

# Gemini
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()

# Supabase (server-side key only)
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_SERVICE_ROLE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()

def _sb() -> Optional[Client]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

supabase: Optional[Client] = _sb()

# Paths / Flask app
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Frontend origin(s)
_frontend_env = (os.getenv("FRONTEND_ORIGIN") or "https://ronit.nbmedia.co.in").rstrip("/")
ALLOWED_ORIGINS = [
    _frontend_env,
    "http://localhost:3000",
    "http://localhost:5173",
]

app = Flask(
    __name__,
    static_url_path="",
    static_folder=str(ROOT_DIR)  # static files allowed (not required for Hostinger)
)

# CORS: allow only your site + localhost for dev
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False, max_age=3600)

# HTTP session with retries
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.35,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["POST", "GET"]),
)
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://",  HTTPAdapter(max_retries=retries))

# =========================
# Helpers
# =========================
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _absolute_url(path: str) -> str:
    """
    Build an absolute URL that respects X-Forwarded-Proto on Render.
    """
    # request.host_url usually like "http://service.onrender.com/"
    base = request.host_url
    proto = request.headers.get("X-Forwarded-Proto")
    if proto and base.startswith("http://") and proto == "https":
        base = base.replace("http://", "https://", 1)
    if not path.startswith("/"):
        path = "/" + path
    return base.rstrip("/") + path

# =========================
# ElevenLabs token helpers
# =========================
# Token cache keyed by participant name: key -> (token, timestamp)
_TOKEN_CACHE: Dict[str, Tuple[Optional[str], datetime]] = {}

def _extract_token(payload: dict) -> Optional[str]:
    for k in ("token", "access_token", "conversation_token"):
        v = payload.get(k)
        if isinstance(v, str) and v:
            return v
    for c in ("conversation", "data", "result"):
        obj = payload.get(c)
        if isinstance(obj, dict):
            for k in ("token", "access_token", "conversation_token"):
                v = obj.get(k)
                if isinstance(v, str) and v:
                    return v
    return None

def _candidate_urls() -> List[str]:
    return [
        os.getenv("ELEVEN_TOKEN_URL") or "",  # if set, try first
        "https://api.elevenlabs.io/v1/convai/conversations",
        "https://api.elevenlabs.io/v1/convai/conversation/token",
        "https://api.elevenlabs.io/v1/convai/conversation",
    ]

def _get_eleven_token(participant_name: Optional[str] = None) -> str:
    if not ELEVEN_API_KEY:
        raise RuntimeError("Missing ELEVEN_API_KEY")
    if not AGENT_ID:
        raise RuntimeError("Missing AGENT_ID or ELEVEN_AGENT_ID")

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_API_KEY,
        "Connection": "keep-alive",
    }
    base_payload = {"agent_id": AGENT_ID}
    if participant_name:
        base_payload["participant_name"] = participant_name  # <- important

    tried = []
    for url in [u for u in _candidate_urls() if u]:
        for method in ("POST", "GET"):
            try:
                resp = session.request(
                    method, url,
                    headers=headers,
                    json=base_payload if method == "POST" else None,
                    params=base_payload if method == "GET" else None,
                    timeout=9
                )
            except requests.RequestException as e:
                tried.append({"url": url, "method": method, "error": f"request_exception:{e.__class__.__name__}"})
                continue

            if resp.status_code < 400:
                try:
                    payload = resp.json()
                except Exception:
                    tried.append({"url": url, "method": method, "status": resp.status_code, "error": "non_json", "text": resp.text[:300]})
                    continue
                token = _extract_token(payload)
                if token:
                    print(f"[TOKEN] OK via {method} {url} (participant_name={participant_name!r})")
                    return token
                tried.append({"url": url, "method": method, "status": resp.status_code, "error": "no_token_in_payload", "payload": payload})
            else:
                try:
                    snippet = resp.json()
                except Exception:
                    snippet = {"raw": resp.text[:300]}
                tried.append({"url": url, "method": method, "status": resp.status_code, "payload": snippet})

    print("[TOKEN] All attempts failed:", tried)
    raise RuntimeError("All token endpoints failed", {"attempts": tried})

# =========================
# Routes
# =========================

@app.get("/")
def root_ok():
    # Keep it simple: backend is alive; frontend is on Hostinger.
    return jsonify({
        "ok": True,
        "service": "AI Voice Coach Backend",
        "time": _now_utc_iso(),
        "frontend": ALLOWED_ORIGINS[0]
    }), 200

@app.get("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "time": _now_utc_iso(),
        "supabase": bool(supabase),
    }), 200

# Optional static serving (not needed for Hostinger, but harmless)
@app.get("/<path:path>")
def static_files(path):
    file_path = ROOT_DIR / path
    if file_path.is_file():
        return send_from_directory(str(ROOT_DIR), path)
    return abort(404)

# Conversation token — accepts ?name= to forward as participant_name
@app.get("/conversation-token")
def conversation_token():
    name = (request.args.get("name") or "").strip()
    cache_key = name or ""
    token, ts = _TOKEN_CACHE.get(cache_key, (None, datetime.min.replace(tzinfo=timezone.utc)))
    if token and (datetime.now(timezone.utc) - ts) < timedelta(seconds=55):
        return jsonify({"token": token})

    try:
        token = _get_eleven_token(name if name else None)
        _TOKEN_CACHE[cache_key] = (token, datetime.now(timezone.utc))
        return jsonify({"token": token})
    except Exception as e:
        details = {} if not isinstance(e, tuple) or len(e) < 2 else e[1]
        msg = str(e if not isinstance(e, tuple) else e[0])
        return jsonify({"error": "ElevenLabs upstream failed", "message": msg, "details": details}), 502

# ===== Supabase-backed profile (name, email, phone, age) =====
@app.post("/profile")
def profile_upsert():
    """
    JSON body:
      { "name": "...", "email": "...", "phone": "...", "age": 25 }
    Upserts into public.profiles on conflict(email).
    """
    if not supabase:
        return jsonify({"error": "Supabase not configured"}), 500

    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    name  = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    phone = (data.get("phone") or "").strip()
    age   = data.get("age", None)

    if not name:
        return jsonify({"error": "Missing name"}), 400
    if not email:
        return jsonify({"error": "Missing email"}), 400

    # coerce age -> int or None
    if age in ("", None):
        age_val = None
    else:
        try:
            age_val = int(age)
        except Exception:
            return jsonify({"error": "age must be an integer"}), 400

    payload = {
        "name": name,
        "email": email,
        "phone": phone or None,
        "age": age_val,
        "updated_at": _now_utc_iso(),
    }

    try:
        supabase.table("profiles").upsert(payload, on_conflict="email").execute()
    except Exception as e:
        return jsonify({"error": f"Supabase upsert failed: {e}"}), 500

    return jsonify({"ok": True})

# ===== Upload & blueprint =====
blueprint_storage: Dict[str, Dict[str, str]] = {}
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "20000"))

@app.post("/upload-session")
def upload_session():
    email = (request.form.get("email") or "").strip()
    transcript = (request.form.get("transcript") or "").strip()
    if not email:
        return jsonify({"error": "Missing email"}), 400
    if not transcript:
        return jsonify({"error": "Missing transcript"}), 400

    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:MAX_TRANSCRIPT_CHARS]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    session_id = str(uuid.uuid4())[:8]

    try:
        (DATA_DIR / f"session_{stamp}_{session_id}.txt").write_text(transcript, encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to save transcript: {e}")

    blueprint = _call_gemini_mindmap_blueprint(transcript)
    blueprint_id = f"{stamp}_{session_id}"
    blueprint_storage[blueprint_id] = {
        "content": blueprint,
        "email": email,
        "created": _now_utc_iso(),
        "session_id": session_id,
    }

    blueprint_link = _absolute_url(f"/blueprint/{blueprint_id}")

    # Send email with link after a short randomized delay
    delay_seconds = random.randint(10, 30)
    threading.Thread(
        target=_delayed_email_with_link,
        args=(email, blueprint_link, delay_seconds),
        daemon=True,
    ).start()

    return jsonify({
        "ok": True,
        "scheduled_in_seconds": delay_seconds,
        "blueprint_id": blueprint_id,
        "blueprint_url": blueprint_link
    })

@app.get("/blueprint/<blueprint_id>")
def view_blueprint(blueprint_id):
    if blueprint_id not in blueprint_storage:
        abort(404, description="Blueprint not found")
    blueprint_data = blueprint_storage[blueprint_id]
    template = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>🧠 Mind Map Blueprint</title>
<style>
body { font-family: Arial, sans-serif; background: linear-gradient(135deg, #667eea, #764ba2); margin: 0; padding: 20px; }
.container { max-width: 1000px; margin: 0 auto; background: white; border-radius: 20px; padding: 30px; }
h1 { text-align: center; color: #333; }
pre { white-space: pre-wrap; word-wrap: break-word; }
.btn { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 20px; margin: 10px; cursor: pointer; }
</style></head><body>
<div class="container">
<h1>🧠 Mind Map Blueprint</h1>
<button class="btn" onclick="window.print()">🖨️ Print</button>
<div style="margin-top: 20px; padding: 15px; background: #f0f8f0; border-radius: 10px;">
<strong>📧 Email:</strong> {{ email }}<br>
<strong>📅 Created:</strong> {{ created }}<br>
<strong>Blueprint:</strong><br><pre>{{ content }}</pre>
</div></div></body></html>'''
    return render_template_string(template, **blueprint_data)

# =========================
# Email helpers
# =========================
def _delayed_email_with_link(email: str, link: str, delay_seconds: int):
    try:
        time.sleep(delay_seconds)
        subject = "🧠 Your Mind Map Blueprint is Ready!"
        body = f"""Hello!

Your conversation analysis is complete. Click the link below to view your Mind Map Blueprint:
{link}

Best regards,
AI Voice Coach Team"""
        ok = _send_email(email, subject, body)
        print(f"[MAIL] Link sent to {email}: {ok}")
    except Exception as e:
        print(f"[ERROR] Email failed for {email}: {e}")

def _send_email(to_email: str, subject: str, body: str) -> bool:
    # 1) SendGrid (preferred)
    if SENDGRID_API_KEY:
        payload = {
            "personalizations": [{
                "to": [{"email": to_email}],
                **({"reply_to": {"email": REPLY_TO}} if REPLY_TO else {})
            }],
            "from": {"email": FROM_EMAIL, "name": FROM_NAME},
            "subject": subject,
            "content": [{"type": "text/plain", "value": body}],
            "mail_settings": {"sandbox_mode": {"enable": SENDGRID_SANDBOX}}
        }
        try:
            r = session.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers={"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type": "application/json"},
                json=payload, timeout=30
            )
            if r.status_code in (200, 202):
                return True
            print(f"[ERROR] SendGrid error: {r.status_code} {r.text[:300]}")
        except Exception as e:
            print(f"[ERROR] SendGrid request failed: {e}")
        # fallthrough to SMTP if configured
    else:
        print(f"[SKIP] No SendGrid key configured. Would have sent to {to_email}")

    # 2) SMTP fallback (optional)
    if SMTP_HOST and SMTP_USER and SMTP_PASSWORD:
        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = f"{FROM_NAME} <{FROM_EMAIL}>"
            msg["To"] = to_email
            if REPLY_TO:
                msg["Reply-To"] = REPLY_TO

            if SMTP_TLS:
                context = ssl.create_default_context()
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                    server.starttls(context=context)
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.sendmail(FROM_EMAIL, [to_email], msg.as_string())
            else:
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.sendmail(FROM_EMAIL, [to_email], msg.as_string())
            return True
        except Exception as e:
            print(f"[ERROR] SMTP send failed: {e}")
            return False

    return False

@app.get("/email/test")
def email_test():
    to = (request.args.get("to") or "").strip()
    if not to:
        return jsonify(error="Provide ?to=email@example.com"), 400
    ok = _send_email(to, "Test from AI Voice Coach", "This is a test email. If you received it, your email setup works.")
    return jsonify(ok=ok)

# =========================
# Gemini helper
# =========================
def _call_gemini_mindmap_blueprint(transcript: str) -> str:
    if not GEMINI_API_KEY:
        return f"CONVERSATION SUMMARY (API Key Missing):\n{transcript[:200]}..."
    prompt = (
        "Create a mind map style summary of this conversation, strictly using markdown formatting like:\n"
        "- **Main Topic**\n  - *Subtopic 1*\n    - Detail 1\n    - Detail 2\n\n"
        f"{transcript}"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = session.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("candidates") and data["candidates"][0]["content"]["parts"]:
            text = data["candidates"][0]["content"]["parts"][0].get("text", "").strip()
            return text or f"CONVERSATION SUMMARY (empty):\n{transcript[:200]}..."
        return f"CONVERSATION SUMMARY (Gemini Empty Response):\n{transcript[:200]}..."
    except Exception as e:
        print(f"[WARN] Gemini API error: {e}")
        return f"CONVERSATION SUMMARY (Gemini API Error):\n{transcript[:200]}..."

# =========================
# Main
# =========================
if __name__ == "__main__":
    print("=== Backend boot ===")
    print("Frontend origin:", ALLOWED_ORIGINS[0])
    print("Agent ID set:", bool(AGENT_ID))
    print("ElevenLabs key set:", bool(ELEVEN_API_KEY))
    if SENDGRID_API_KEY:
        print("SendGrid: ENABLED (sandbox:", SENDGRID_SANDBOX, ")")
    else:
        print("SendGrid: not configured; will try SMTP if provided.")
    if SMTP_HOST:
        print(f"SMTP: {SMTP_HOST}:{SMTP_PORT}, user={SMTP_USER!r}, TLS={SMTP_TLS}")
    print("Supabase configured:", bool(supabase))
    app.run(host="0.0.0.0", port=PORT, debug=False)
