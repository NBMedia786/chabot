# # # app.py — email + ElevenLabs token (with participant_name) + Supabase profiles (name/email/phone/age)

import os, random, threading, time, uuid, smtplib, ssl, mimetypes, base64
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

from flask import Flask, jsonify, send_from_directory, abort, request, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter, Retry

# ── Supabase client
# pip install supabase
from supabase import create_client, Client  # type: ignore

load_dotenv()

# ===== Core config =====
ELEVEN_API_KEY = (os.getenv("ELEVEN_API_KEY") or "").strip()
AGENT_ID       = (os.getenv("AGENT_ID") or os.getenv("ELEVEN_AGENT_ID") or "").strip()
PORT           = int(os.getenv("PORT", "5000"))

# ===== Email config =====
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

# ===== Gemini =====
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()

# ===== Supabase =====
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_SERVICE_ROLE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()

def _sb() -> Optional[Client]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

supabase: Optional[Client] = _sb()

# ===== Paths / app =====
ROOT_DIR   = Path(__file__).parent
# PUBLIC_DIR = ROOT_DIR / "public"
DATA_DIR   = ROOT_DIR / "data"
# PUBLIC_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_url_path="", static_folder=str(ROOT_DIR))
# ---- CORS: allow your Hostinger frontend + local dev ----
_frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")  # e.g., https://yourdomain.com
CORS(app, resources={r"/*": {"origins": [_frontend_origin, "http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5500"]}}, supports_credentials=True)

# ===== HTTP session with retries =====
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.35,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["POST","GET"]),
)
session.mount("https://", HTTPAdapter(max_retries=retries))
session.mount("http://",  HTTPAdapter(max_retries=retries))

# ===== Token cache keyed by participant name =====
# key = participant_name or ""  -> (token, timestamp)
_TOKEN_CACHE: Dict[str, Tuple[Optional[str], datetime]] = {}

def _extract_token(payload: dict) -> Optional[str]:
    for k in ("token","access_token","conversation_token"):
        v = payload.get(k)
        if isinstance(v, str) and v:
            return v
    for c in ("conversation","data","result"):
        obj = payload.get(c)
        if isinstance(obj, dict):
            for k in ("token","access_token","conversation_token"):
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
    if not ELEVEN_API_KEY: raise RuntimeError("Missing ELEVEN_API_KEY")
    if not AGENT_ID:       raise RuntimeError("Missing AGENT_ID or ELEVEN_AGENT_ID")

    headers = {
        "Accept":"application/json",
        "Content-Type":"application/json",
        "xi-api-key":ELEVEN_API_KEY,
        "Connection":"keep-alive",
    }
    # Forward the user's name as participant_name for personalization
    base_payload = {"agent_id": AGENT_ID}
    if participant_name:
        base_payload["participant_name"] = participant_name

    tried = []
    for url in [u for u in _candidate_urls() if u]:
        for method in ("POST","GET"):
            try:
                resp = session.request(
                    method, url,
                    headers=headers,
                    json=base_payload if method=="POST" else None,
                    params=base_payload if method=="GET"  else None,
                    timeout=9
                )
            except requests.RequestException as e:
                tried.append({"url":url,"method":method,"error":f"request_exception:{e.__class__.__name__}"})
                continue

            if resp.status_code < 400:
                try:
                    payload = resp.json()
                except Exception:
                    tried.append({"url":url,"method":method,"status":resp.status_code,"error":"non_json","text":resp.text[:300]})
                    continue
                token = _extract_token(payload)
                if token:
                    print(f"[TOKEN] OK via {method} {url} (participant_name={participant_name!r})")
                    return token
                tried.append({"url":url,"method":method,"status":resp.status_code,"error":"no_token_in_payload","payload":payload})
            else:
                try:
                    snippet = resp.json()
                except Exception:
                    snippet = {"raw": resp.text[:300]}
                tried.append({"url":url,"method":method,"status":resp.status_code,"payload":snippet})

    print("[TOKEN] All attempts failed:", tried)
    raise RuntimeError("All token endpoints failed", {"attempts": tried})

# ===== Routes =====
@app.get("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "time": datetime.utcnow().isoformat()+"Z",
        "supabase": bool(supabase),
    })

@app.get("/")
def index():
    return send_from_directory( "index.html")
# def index():
#     return send_from_directory(str(PUBLIC_DIR), "index.html")

@app.get("/<path:path>")
def static_files(path):
    # return send_from_directory(path)
    return send_from_directory(str(ROOT_DIR), path)

# Conversation token — accepts ?name= to forward as participant_name
@app.get("/conversation-token")
def conversation_token():
    name = (request.args.get("name") or "").strip()
    cache_key = name or ""

    token, ts = _TOKEN_CACHE.get(cache_key, (None, datetime.min))
    if token and (datetime.utcnow() - ts) < timedelta(seconds=55):
        return jsonify({"token": token})

    try:
        token = _get_eleven_token(name if name else None)
        _TOKEN_CACHE[cache_key] = (token, datetime.utcnow())
        return jsonify({"token": token})
    except Exception as e:
        details = {} if not isinstance(e, tuple) or len(e) < 2 else e[1]
        return jsonify({"error":"ElevenLabs upstream failed","message":str(e if not isinstance(e, tuple) else e[0]),"details":details}), 502

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
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }

    try:
        supabase.table("profiles").upsert(payload, on_conflict="email").execute()
    except Exception as e:
        return jsonify({"error": f"Supabase upsert failed: {e}"}), 500

    return jsonify({"ok": True})

# ===== Upload & blueprint (your existing behavior) =====
blueprint_storage = {}

@app.post("/upload-session")
def upload_session():
    email = (request.form.get("email") or "").strip()
    transcript = (request.form.get("transcript") or "").strip()
    if not email:      return jsonify({"error":"Missing email"}), 400
    if not transcript: return jsonify({"error":"Missing transcript"}), 400

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
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
        "created": datetime.utcnow().isoformat(),
        "session_id": session_id,
    }
    blueprint_link = f"{request.host_url}blueprint/{blueprint_id}"

    delay_seconds = random.randint(10, 30)
    threading.Thread(
        target=_delayed_email_with_link,
        args=(email, blueprint_link, delay_seconds),
        daemon=True,
    ).start()

    return jsonify({"ok": True, "scheduled_in_seconds": delay_seconds})

@app.get("/blueprint/<blueprint_id>")
def view_blueprint(blueprint_id):
    if blueprint_id not in blueprint_storage:
        abort(404, description="Blueprint not found")
    blueprint_data = blueprint_storage[blueprint_id]
    template = '''<!DOCTYPE html>
<html><head><title>🧠 Mind Map Blueprint</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
body { font-family: Arial; background: linear-gradient(135deg, #667eea, #764ba2); margin: 0; padding: 20px; }
.container { max-width: 1000px; margin: 0 auto; background: white; border-radius: 20px; padding: 30px; }
h1 { text-align: center; color: #333; }
#mindmap { width: 100%; height: 500px; border: 2px solid #ddd; border-radius: 10px; background: #fafafa; }
.btn { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 20px; margin: 10px; cursor: pointer; }
</style></head><body>
<div class="container">
<h1>🧠 Mind Map Blueprint</h1>
<button class="btn" onclick="window.print()">🖨️ Print</button>
<div id="mindmap"></div>
<div style="margin-top: 20px; padding: 15px; background: #f0f8f0; border-radius: 10px;">
<strong>📧 Email:</strong> {{ email }}<br>
<strong>📅 Created:</strong> {{ created }}<br>
<strong>Blueprint:</strong><br><pre>{{ content }}</pre>
</div></div></body></html>'''
    return render_template_string(template, **blueprint_data)

# ===== Email helpers =====
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
    # 1) SendGrid
    if SENDGRID_API_KEY:
        payload = {
            "personalizations": [{
                "to": [{"email": to_email}],
                **({"reply_to": {"email": REPLY_TO}} if REPLY_TO else {})
            }],
            "from": {"email": FROM_EMAIL, "name": FROM_NAME},
            "subject": subject,
            "content": [{"type":"text/plain","value": body}],
            "mail_settings": {"sandbox_mode": {"enable": SENDGRID_SANDBOX}}
        }
        r = session.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type":"application/json"},
            json=payload, timeout=30
        )
        if r.status_code in (200, 202):
            return True
        print(f"[ERROR] SendGrid error: {r.status_code} {r.text[:300]}")
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

# ===== Email test route =====
@app.get("/email/test")
def email_test():
    to = (request.args.get("to") or "").strip()
    if not to:
        return jsonify(error="Provide ?to=email@example.com"), 400
    ok = _send_email(to, "Test from AI Voice Coach", "This is a test email. If you received it, your email setup works.")
    return jsonify(ok=ok)

# ===== Gemini helper =====
def _call_gemini_mindmap_blueprint(transcript: str) -> str:
    if not GEMINI_API_KEY:
        return f"CONVERSATION SUMMARY (API Key Missing):\n{transcript[:200]}..."
    prompt = (
        "Create a mind map style summary of this conversation, strictly using markdown formatting like:\n"
        "- **Main Topic**\n  - *Subtopic 1*\n    - Detail 1\n    - Detail 2\n\n"
        f"{transcript}"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents":[{"parts":[{"text":prompt}]}]}
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


@app.route("/healthz", methods=["GET"])
def healthcheck():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat() + "Z"}), 200

# ===== Main =====


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    print("=== Email config ===")
    print("FROM_EMAIL:", FROM_EMAIL)
    print("FROM_NAME:", FROM_NAME)
    print("REPLY_TO:", REPLY_TO)
    print("Using SendGrid:", bool(SENDGRID_API_KEY))
    print("Supabase:", bool(SUPABASE_URL and SUPABASE_KEY))
    print("Agent ID set:", bool(AGENT_ID))
    print("ElevenLabs key set:", bool(ELEVEN_API_KEY))
    print("Frontend allowed origin:", os.getenv("FRONTEND_ORIGIN"))
    app.run(host="0.0.0.0", port=port, debug=debug)
