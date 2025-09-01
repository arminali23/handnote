# webapp/server/app.py
import os, base64
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==================== Model ====================
class SmallCNN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "emnist_letters_cnn.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if torch.backends.mps.is_available() else "cpu"))
_model = None

def load_model():
    global _model
    if _model is None:
        m = SmallCNN(26).to(DEVICE)
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        m.load_state_dict(ckpt["state_dict"], strict=True)
        m.eval()
        _model = m
        print("EMNIST model loaded on", DEVICE)
    return _model

# ==================== Preprocess (MNIST-style) ====================
def mnist_preprocess_from_rgb(rgb_np: np.ndarray, margin=16) -> np.ndarray | None:
    """
    Input: RGB (H x W x 3) numpy uint8 (black bg, white strokes).
    Output: 28x28 uint8 (black BG, white FG) or None if empty.
    """
    if rgb_np.ndim != 3 or rgb_np.shape[2] != 3:
        raise ValueError("Expected RGB uint8 image")

    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY)  # keep as drawn

    ys, xs = np.where(bw > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    y1 = max(0, y1 - margin); x1 = max(0, x1 - margin)
    y2 = min(bw.shape[0]-1, y2 + margin); x2 = min(bw.shape[1]-1, x2 + margin)
    crop = bw[y1:y2+1, x1:x2+1]

    # enforce black BG / white FG
    if crop.mean() >= 90:
        crop = 255 - crop

    # deskew
    m = cv2.moments(crop)
    if abs(m['mu02']) > 1e-2:
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5 * skew * crop.shape[0]], [0, 1, 0]])
        crop = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                              flags=cv2.INTER_LINEAR, borderValue=0)

    # scale longest side to 20
    h, w = crop.shape
    if h > w:
        new_h = 20; new_w = max(1, int(round(w * (20.0 / h))))
    else:
        new_w = 20; new_h = max(1, int(round(h * (20.0 / w))))
    resized20 = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # pad to 28 + mass center
    canvas28 = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas28[y_off:y_off+new_h, x_off:x_off+new_w] = resized20

    m2 = cv2.moments(canvas28)
    if m2['m00'] > 0:
        cx = int(m2['m10'] / m2['m00']); cy = int(m2['m01'] / m2['m00'])
        shiftx = int(round(14 - cx)); shifty = int(round(14 - cy))
        M2 = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        canvas28 = cv2.warpAffine(canvas28, M2, (28, 28),
                                  flags=cv2.INTER_LINEAR, borderValue=0)
    return canvas28

def predict_emnist_averaged(img28_uint8: np.ndarray):
    """TTA (rotations + light morphology). Returns (pred_char, conf, top3 list)."""
    base = img28_uint8 if img28_uint8.mean() < 90 else (255 - img28_uint8)
    variants = [
        base,
        cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.rotate(base, cv2.ROTATE_180),
        cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE),
    ]
    k = np.ones((2,2), np.uint8)
    variants += [cv2.dilate(base, k, iterations=1),
                 cv2.erode(base, k, iterations=1)]
    with torch.no_grad():
        model = load_model()
        probs_sum = None
        for v in variants:
            x = torch.from_numpy(v).float().div(255.0)
            x = (x - 0.1307) / 0.3081
            x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)
            p = torch.softmax(model(x), dim=1).cpu().numpy()
            probs_sum = p if probs_sum is None else (probs_sum + p)
    probs = (probs_sum / len(variants)).ravel()
    idx = int(probs.argmax())
    pred_char = chr(ord('A') + idx)
    top3_idx = probs.argsort()[::-1][:3]
    top3 = [{"char": chr(ord('A') + int(i)), "prob": float(probs[i])} for i in top3_idx]
    return pred_char, float(probs[idx]), top3

# ==================== API ====================
class PredictBody(BaseModel):
    # data URL (e.g. "data:image/png;base64,...") of canvas (black bg, white strokes)
    image: str

class SaveBody(BaseModel):
    text: str

NOTES_DIR = os.path.join(HERE, "notes")
os.makedirs(NOTES_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/api/health")
def health():
    load_model()
    return {"ok": True}

@app.post("/api/predict")
def api_predict(body: PredictBody):
    try:
        header, b64data = body.image.split(",", 1)
        png_bytes = base64.b64decode(b64data)
        arr = np.frombuffer(png_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)  # RGBA/RGB

        # normalize to RGB
        if img.ndim == 3 and img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            return {"ok": False, "message": "Invalid image"}

        img28 = mnist_preprocess_from_rgb(rgb)
        if img28 is None:
            return {"ok": False, "message": "No ink found."}

        ch, conf, top3 = predict_emnist_averaged(img28)
        return {"ok": True, "char": ch, "conf": conf, "top3": top3}
    except Exception as e:
        return {"ok": False, "message": str(e)}

@app.post("/api/save_text")
def api_save_text(body: SaveBody):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(NOTES_DIR, f"note_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body.text)
    return {"ok": True, "path": path}