import os
import cv2
import numpy as np
from datetime import datetime

import mediapipe as mp

import torch
import torch.nn as nn


# ========================== EMNIST MODEL ==========================
class SmallCNN(nn.Module):
    """Compact CNN for EMNIST Letters (A–Z)."""
    def __init__(self, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # -> 14x14

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # -> 7x7

            nn.Flatten(),
            nn.Linear(64*7*7, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


EMNIST_MODEL_PATH = "emnist_letters_cnn.pt"
_emnist_model = None
_emnist_device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

def load_emnist_model():
    """Lazy-load the EMNIST model once and keep it in memory."""
    global _emnist_model
    if _emnist_model is None:
        m = SmallCNN(num_classes=26).to(_emnist_device)
        ckpt = torch.load(EMNIST_MODEL_PATH, map_location=_emnist_device)
        m.load_state_dict(ckpt["state_dict"], strict=True)
        m.eval()
        _emnist_model = m
        print("EMNIST model loaded on", _emnist_device)
    return _emnist_model


# ========================== DRAW/APP SETTINGS ==========================
DRAW_THICKNESS   = 10
PINCH_THRESHOLD  = 0.07     # pinch (thumb-index) threshold
SMOOTHING        = 0.4      # fingertip EMA (0..1)
SAVE_DIR         = "samples"
NOTES_DIR        = "notes"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)

# live text buffer (builds your note on screen)
TEXT_BUFFER = []  # list of characters

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ========================== HELPERS ==========================
def norm_dist(a, b):
    """Normalized distance between two Mediapipe landmarks in [0..1] coords."""
    return np.linalg.norm(np.array([a.x - b.x, a.y - b.y]))

def get_text_string():
    """Join the TEXT_BUFFER into a single string (for saving)."""
    return "".join(TEXT_BUFFER)

def save_textbuffer_to_file():
    """Save current TEXT_BUFFER to a timestamped .txt in NOTES_DIR."""
    content = get_text_string()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(NOTES_DIR, f"note_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def draw_textbuffer_overlay(img):
    """
    Render the live text buffer at the bottom-left of the window.
    Shows at most the last ~2 lines (for readability).
    """
    h, w = img.shape[:2]
    text = get_text_string()
    # split into lines so \n is respected; show the last two lines
    lines = text.split("\n")
    last_lines = lines[-2:] if len(lines) >= 2 else lines
    y = h - 40  # start a bit above the bottom
    for line in last_lines:
        cv2.putText(img, f"Text: {line}" if line else "Text:",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (50, 230, 50), 2, cv2.LINE_AA)
        y += 28  # line spacing


# ========================== PREPROCESS & SAVE (MNIST-style) ==========================
def preprocess_and_save(img_bgr, save_dir=SAVE_DIR, preview_size=1024, margin=16):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(bw > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # safe bbox with margin
    y1 = max(0, y1 - margin); x1 = max(0, x1 - margin)
    y2 = min(bw.shape[0]-1, y2 + margin); x2 = min(bw.shape[1]-1, x2 + margin)
    crop = bw[y1:y2+1, x1:x2+1]

    # enforce black BG / white FG
    if crop.mean() >= 90:
        crop = 255 - crop

    # deskew (classic MNIST trick)
    m = cv2.moments(crop)
    if abs(m['mu02']) > 1e-2:
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5 * skew * crop.shape[0]], [0, 1, 0]])
        crop = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                              flags=cv2.INTER_LINEAR, borderValue=0)

    # scale to 20 on the longest side
    h, w = crop.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))
    else:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))
    resized20 = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # pad to 28x28
    canvas28 = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas28[y_off:y_off+new_h, x_off:x_off+new_w] = resized20

    # re-center by mass
    m2 = cv2.moments(canvas28)
    if m2['m00'] > 0:
        cx = int(m2['m10'] / m2['m00'])
        cy = int(m2['m01'] / m2['m00'])
        shiftx = int(round(14 - cx))
        shifty = int(round(14 - cy))
        M2 = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        canvas28 = cv2.warpAffine(canvas28, M2, (28, 28),
                                  flags=cv2.INTER_LINEAR, borderValue=0)

    # save preview (big) and model image (28x28)
    preview = cv2.resize(canvas28, (preview_size, preview_size), interpolation=cv2.INTER_NEAREST)

    base = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path_crop  = os.path.join(save_dir, f"{base}_crop.png")    # stores final 28x28
    path_prev  = os.path.join(save_dir, f"{base}_preview.png")
    path_model = os.path.join(save_dir, f"{base}_28x28.png")

    cv2.imwrite(path_crop,  canvas28)
    cv2.imwrite(path_prev,  preview)
    cv2.imwrite(path_model, canvas28)

    return {"crop": path_crop, "preview": path_prev, "model": path_model, "side": 28}


# ========================== PREDICT (TTA) ==========================
def predict_emnist_from_png(png_path, topk=3, show_debug=False):
  
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (28, 28):
        print("Invalid 28x28 image path:", png_path)
        return None

    base = img if img.mean() < 90 else (255 - img)

    variants = [base,
                cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE),
                cv2.rotate(base, cv2.ROTATE_180),
                cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE)]
    k = np.ones((2, 2), np.uint8)
    variants += [cv2.dilate(base, k, iterations=1),
                 cv2.erode(base, k, iterations=1)]

    with torch.no_grad():
        model = load_emnist_model()
        probs_sum = None
        for v in variants:
            x = torch.from_numpy(v).float().div(255.0)
            x = (x - 0.1307) / 0.3081
            x = x.unsqueeze(0).unsqueeze(0).to(_emnist_device)  # (1,1,28,28)
            logits = model(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs_sum = p if probs_sum is None else (probs_sum + p)

        probs = (probs_sum / len(variants)).ravel()
        idx = int(probs.argmax())
        pred_char = chr(ord('A') + idx)

        if topk > 1:
            top_idx = probs.argsort()[::-1][:topk]
            tops = [(chr(ord('A') + int(i)), float(probs[i])) for i in top_idx]
            print("Top-{}: {}".format(topk, tops))

        print(f"Prediction: {pred_char} (conf={float(probs[idx]):.2f})")
        return pred_char


# ========================== MAIN LOOP (DRAW + TEXT) ==========================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera did not open.")
        return

    ok, frame = cap.read()
    if not ok:
        print("Could not read frame.")
        return

    H, W = frame.shape[:2]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    last_point = None
    filtered_point = None
    drawing = False
    color = (255, 255, 255)

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            index_tip_px = None
            pinch_now = False

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                idx_tip = hand.landmark[8]   # index fingertip
                thb_tip = hand.landmark[4]   # thumb tip

                d = norm_dist(idx_tip, thb_tip)
                pinch_now = d < PINCH_THRESHOLD

                ix, iy = int(idx_tip.x * W), int(idx_tip.y * H)
                index_tip_px = (ix, iy)

                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # fingertip smoothing + draw
            if index_tip_px is not None:
                if filtered_point is None:
                    filtered_point = np.array(index_tip_px, dtype=np.float32)
                else:
                    filtered_point = (
                        SMOOTHING * filtered_point
                        + (1 - SMOOTHING) * np.array(index_tip_px, dtype=np.float32)
                    )

                ip = tuple(filtered_point.astype(int))

                if pinch_now and not drawing:
                    drawing = True
                    last_point = ip
                elif not pinch_now and drawing:
                    drawing = False
                    last_point = None

                if drawing and last_point is not None:
                    cv2.line(canvas, last_point, ip, color, DRAW_THICKNESS, cv2.LINE_AA)
                    last_point = ip

            blend = cv2.addWeighted(frame, 0.7, canvas, 0.9, 0)

            # UI + text buffer overlay
            cv2.putText(
                blend,
                "Pinch=draw   N=save   E=predict->Text   Space/Enter/Backspace   S=save.txt   C=clear   Q=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (20, 220, 20), 2, cv2.LINE_AA
            )
            status_color = (0, 255, 0) if drawing else (0, 0, 255)
            cv2.rectangle(blend, (10, 45), (30, 65), status_color, -1)

            # draw current text at bottom
            draw_textbuffer_overlay(blend)

            cv2.imshow("Air-Draw Notepad", blend)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):      # Quit
                break

            elif key == ord('c'):    # Clear canvas
                canvas[:] = 0

            elif key == ord('n'):    # Save current glyph (MNIST-style)
                out = preprocess_and_save(canvas)
                if out:
                    print(f"Saved 28x28: {out['model']}")
                    print(f"Saved preview: {out['preview']}")
                    canvas[:] = 0
                else:
                    print("No ink to save.")

            elif key == ord('e'):    # Predict last saved glyph -> append to TEXT_BUFFER
                import glob
                paths = sorted(
                    glob.glob(os.path.join(SAVE_DIR, "*_28x28.png")),
                    key=os.path.getmtime
                )
                if not paths:
                    print("Save a glyph with 'N' first.")
                else:
                    last28 = paths[-1]
                    pred = predict_emnist_from_png(last28, topk=3, show_debug=False)
                    if pred is not None:
                        TEXT_BUFFER.append(pred)

            elif key == 32:          # Space
                TEXT_BUFFER.append(" ")

            elif key == 13:          # Enter (newline)
                TEXT_BUFFER.append("\n")

            elif key == 8:           # Backspace
                if TEXT_BUFFER:
                    TEXT_BUFFER.pop()

            elif key == ord('s'):    # Save current text to notes/*.txt
                path = save_textbuffer_to_file()
                print("Saved note:", path)

    cap.release()
    cv2.destroyAllWindows()


# keep mp_hands available after defs
mp_hands = mp.solutions.hands

if __name__ == "__main__":
    main()
