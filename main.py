"""
Quantum Road Scanner (QRS) v2.4
───────────────────────────────────────────────────────────────────────────────
• Async-safe rate-limiter (asyncio.Lock)
• Proper Hadamard gates per wire
• GPU AES stub returns bytes
• Guaranteed camera release on error
• ENV-var fallback for OpenAI key
───────────────────────────────────────────────────────────────────────────────
pip install aiohttp httpx psutil pennylane opencv-python pillow numpy aiosqlite
"""
from __future__ import annotations

# ─── Stdlib ───────────────────────────────────────────────────────────────────
import asyncio
import json
import logging
import os
import secrets
import threading
import time
from base64 import b64decode, b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, List, Optional

# ─── Third-party ──────────────────────────────────────────────────────────────
import httpx
import numpy as np
import psutil
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.simpledialog as simpledialog
import aiosqlite as qrsdb
import cv2
import pennylane as qml
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

try:
    import cupy as cp  # GPU AES placeholder
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

# ─── Config ───────────────────────────────────────────────────────────────────
CROP_TOP, CROP_BOTTOM = 0.6, 0.9
CROP_LEFT, CROP_RIGHT = 0.3, 0.7
GPT_VECTOR_TEMPERATURE:  Final = 0.4
GPT_COMPLETION_TEMPERATURE: Final = 0.6
PROMPT_WORD_LIMIT: Final = 300
OPENAI_TIMEOUT: Final = 20
COOLDOWN_SECONDS: Final = 5

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

_rate_lock = asyncio.Lock()
_last_call: float = 0.0

# ═════════════════════════════════════════════════════════════════════════════
# Crypto
# ═════════════════════════════════════════════════════════════════════════════
class AESGCMCrypto:
    """File-backed AES-128-GCM helper."""
    def __init__(self, key_path: str = "~/.cache/qrs_master_key.bin") -> None:
        self.key_path = os.path.expanduser(key_path)
        if not os.path.exists(self.key_path):
            self.key: bytes = AESGCM.generate_key(bit_length=128)
            os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
            with open(self.key_path, "wb") as f:
                f.write(self.key)
            os.chmod(self.key_path, 0o600)
        else:
            with open(self.key_path, "rb") as f:
                self.key = f.read()
            if len(self.key) != 16:
                raise ValueError("Invalid AES-128 key length.")
        self._aes = AESGCM(self.key)

    def encrypt(self, text: str) -> bytes:
        nonce = secrets.token_bytes(12)
        cipher = self._aes.encrypt(nonce, text.encode(), None)
        return b64encode(nonce + cipher)

    def decrypt(self, token: bytes) -> str:
        raw = b64decode(token)
        return self._aes.decrypt(raw[:12], raw[12:], None).decode()

def gpu_encrypt(data: bytes) -> bytes:
    """Future hook: off-load to GPU when cupy is present."""
    if not GPU_AVAILABLE or not data:
        return data
    buf = cp.asarray(memoryview(data))
    return bytes(cp.asnumpy(buf))

crypto = AESGCMCrypto()

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
def get_cpu_usage() -> float:
    try:  # psutil is non-blocking here
        return psutil.cpu_percent(interval=None)
    except Exception as exc:
        logging.error("CPU usage fetch error: %s", exc)
        return 0.0

class VideoCaptureCtx:
    """Context-managed cv2.VideoCapture for safe release."""
    def __init__(self, index: int = 0) -> None:
        self._cap = cv2.VideoCapture(index, cv2.CAP_ANY)

    def __enter__(self):
        if not self._cap.isOpened():
            raise RuntimeError("Camera unavailable.")
        return self._cap

    def __exit__(self, exc_type, exc, tb):
        self._cap.release()

def camera_color_vector() -> List[int]:
    with VideoCaptureCtx() as cam:
        ok, frame = cam.read()
    if not ok:
        raise RuntimeError("Camera capture failed.")
    h, w, _ = frame.shape
    crop = frame[int(h*CROP_TOP):int(h*CROP_BOTTOM),
                 int(w*CROP_LEFT):int(w*CROP_RIGHT)]
    b, g, r = map(int, crop.mean(axis=(0, 1)))
    return [r, g, b]

# ═════════════════════════════════════════════════════════════════════════════
# Networking helpers
# ═════════════════════════════════════════════════════════════════════════════
async def _rate_limited_post(url: str, headers: dict, payload: dict) -> dict:
    global _last_call
    async with _rate_lock:
        delay = max(0, COOLDOWN_SECONDS - (time.time() - _last_call))
        if delay:
            await asyncio.sleep(delay)
        _last_call = time.time()

    async with httpx.AsyncClient(timeout=OPENAI_TIMEOUT) as client:
        resp = await client.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()

async def gpt_color_vector(api_key: str) -> List[int]:
    prompt = ("Return ONLY a Python list of three integers (0-255) "
              "representing the average road-surface RGB color.")
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": GPT_VECTOR_TEMPERATURE,
    }
    try:
        data = await _rate_limited_post(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {api_key}",
             "Content-Type": "application/json"},
            payload)
        content = data["choices"][0]["message"]["content"].strip()
        # Robust parse
        for parser in (json.loads, lambda s: eval(s, {})):
            try:
                return list(map(int, parser(content)))
            except Exception:
                pass
        raise ValueError("Un-parseable GPT vector.")
    except Exception as exc:
        logging.error("GPT color vector error: %s", exc)
        return [128, 128, 128]

async def run_openai_completion(prompt: str, api_key: str) -> str:
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": GPT_COMPLETION_TEMPERATURE,
    }
    try:
        data = await _rate_limited_post(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {api_key}",
             "Content-Type": "application/json"},
            payload)
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logging.error("OpenAI completion error: %s", exc)
        return "Error: Unable to retrieve completion."

# ═════════════════════════════════════════════════════════════════════════════
# Quantum
# ═════════════════════════════════════════════════════════════════════════════
dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def _quantum_rgb_circuit_node(r: float, g: float, b: float, c: float):
    for w in (0, 1, 2):
        qml.Hadamard(wires=w)
    qml.RX(np.pi * r * c, wires=0)
    qml.RY(np.pi * g * c, wires=1)
    qml.RZ(np.pi * b * c, wires=2)
    qml.CRX(np.pi * r, wires=[0, 1])
    qml.CRY(np.pi * g, wires=[1, 2])
    qml.CRZ(np.pi * b, wires=[2, 0])
    if c > 0.7:
        qml.Toffoli(wires=[0, 1, 2])
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    return qml.probs(wires=[0, 1, 2])

def quantum_rgb(cpu_pct: float, rgb: List[int]) -> List[float]:
    c = cpu_pct / 100.0
    r, g, b = (x / 255.0 for x in rgb)
    return _quantum_rgb_circuit_node(r, g, b, c).tolist()

# ═════════════════════════════════════════════════════════════════════════════
# Database
# ═════════════════════════════════════════════════════════════════════════════
DB_PATH = "qrs_db.db"
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS scans (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT,
    rgb      BLOB,
    cpu      REAL,
    quantum  BLOB,
    vehicle  TEXT,
    route    TEXT,
    prompt   BLOB,
    response BLOB
);
CREATE INDEX IF NOT EXISTS idx_ts ON scans(ts);
"""

async def init_db() -> None:
    async with qrsdb.connect(DB_PATH) as db:
        await db.executescript(CREATE_SQL)
        await db.commit()

# ═════════════════════════════════════════════════════════════════════════════
# Core engine
# ═════════════════════════════════════════════════════════════════════════════
async def scan_and_log(api_key: str, vehicle: str, route_task: str,
                       use_camera: bool) -> str:
    await init_db()

    cpu = get_cpu_usage()
    try:
        rgb = camera_color_vector() if use_camera else await gpt_color_vector(api_key)
    except Exception as exc:
        logging.warning("Camera failure (%s); using GPT fallback.", exc)
        rgb = await gpt_color_vector(api_key)

    probs = quantum_rgb(cpu, rgb)

    prompt = f"""
You are QRS, the Quantum Road Scanner assistant.
RULES:
 1. Prioritize {vehicle}-specific hazards.
 2. Use RGB vector to detect surface anomalies.
 3. Interpret CPU load ({cpu:.2f} %) as urgency scale.
 4. Factor quantum state probabilities {probs} for environmental uncertainty.
 5. Predict hazards for the NEXT-3 route segments.
INPUT:
  Vehicle  : {vehicle}
  Route    : {route_task}
  RGB      : {rgb}
  Quantum  : {probs}
GOALS:
  [1] Risk level (Low/Med/High)
  [2] Top-3 hazards
  [3] Quantum drift insight
  [4] NEXT-3 segment predictions
  [5] Tactical advice
Limit to ≤ {PROMPT_WORD_LIMIT} words.
""".strip()

    response = await run_openai_completion(prompt, api_key)

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    async with qrsdb.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO scans (ts,rgb,cpu,quantum,vehicle,route,prompt,response) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                ts,
                crypto.encrypt(json.dumps(rgb)),
                cpu,
                crypto.encrypt(json.dumps(probs)),
                vehicle,
                route_task,
                crypto.encrypt(prompt),
                crypto.encrypt(response),
            ),
        )
        await db.commit()
    return response

# ═════════════════════════════════════════════════════════════════════════════
# GUI
# ═════════════════════════════════════════════════════════════════════════════
class QRSScannerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Quantum Road Scanner (QRS v2.4)")
        self.geometry("820x720")

        tk.Label(self, text="Quantum Road Scanner",
                 font=("Helvetica", 18, "bold")).pack(pady=8)

        frm = tk.Frame(self); frm.pack(pady=4)

        tk.Label(frm, text="Vehicle Type:").grid(row=0, column=0, sticky="e")
        self.vehicle = tk.StringVar(value="motorcycle")
        tk.OptionMenu(frm, self.vehicle,
                      "motorcycle", "car", "truck", "bicycle").grid(row=0, column=1)

        tk.Label(frm, text="Destination / Task:").grid(row=1, column=0, sticky="e")
        self.route = tk.Entry(frm, width=48)
        self.route.grid(row=1, column=1, pady=3)

        self.cam_var = tk.BooleanVar(value=False)
        tk.Checkbutton(frm, text="Use camera for RGB",
                       variable=self.cam_var).grid(row=2, column=1, sticky="w")

        self.start_btn = tk.Button(self, text="Start Scan",
                                   font=("Helvetica", 14), command=self.start_scan)
        self.start_btn.pack(pady=8)

        self.status = tk.StringVar(value="Idle.")
        tk.Label(self, textvariable=self.status).pack()

        self.result_text = tk.Text(self, height=28, width=98, wrap="word")
        self.result_text.pack(padx=8, pady=8)

        menu = tk.Menu(self)
        menu.add_command(label="Set OpenAI API Key", command=self.prompt_api_key)
        self.config(menu=menu)

        self.api_key_path = "~/.cache/qrs_encrypted_api_key.bin"

    # ─── API key ──────────────────────────────────────────────────────────────
    def prompt_api_key(self) -> None:
        key = simpledialog.askstring("API Key",
                                     "Enter your OpenAI API Key:", show='*')
        if key:
            with open(os.path.expanduser(self.api_key_path), "wb") as f:
                f.write(crypto.encrypt(key))
            self.status.set("API key saved.")

    def load_api_key(self) -> Optional[str]:
        try:
            with open(os.path.expanduser(self.api_key_path), "rb") as f:
                return crypto.decrypt(f.read())
        except Exception:
            return os.getenv("OPENAI_API_KEY")

    # ─── Scan ────────────────────────────────────────────────────────────────
    def start_scan(self) -> None:
        api = self.load_api_key()
        if not api:
            messagebox.showerror("Error", "API key missing or corrupted.")
            return

        self.start_btn.config(state="disabled")
        self.status.set("Scanning...")

        threading.Thread(
            target=self._thread_worker,
            args=(api, self.vehicle.get(),
                  self.route.get().strip() or "unspecified",
                  self.cam_var.get()),
            daemon=True
        ).start()

    def _thread_worker(self, api: str, vehicle: str,
                       route_task: str, use_cam: bool) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                scan_and_log(api, vehicle, route_task, use_cam))
            self.after(0, lambda: self._update_ui(result))
        except Exception as exc:
            logging.exception(exc)
            self.after(0, lambda: messagebox.showerror("Error", str(exc)))
        finally:
            self.after(0, self._reset_ui)

    def _update_ui(self, text: str) -> None:
        ts_local = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.result_text.insert(
            tk.END, f"\n─── Scan Result ({ts_local}) ───\n{text}\n")
        self.result_text.see(tk.END)

    def _reset_ui(self) -> None:
        self.status.set("Idle. Ready for next scan.")
        self.start_btn.config(state="normal")

# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    QRSScannerApp().mainloop()
