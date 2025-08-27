import cv2
import os
import numpy as np
import mediapipe as mp
from datetime import datetime


DRAW_THICKNESS = 6
PINCH_THRESHOLD = 0.07
SMOOTHING = 0.4
SAVE_DIR = "samples"
os.makedirs(SAVE_DIR,exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def norm_dist(a,b): 
    return np.linalg.norm(np.array([a.x-b.x,a.y-b.y]))

def preprocess_and_save(img, save_dir=SAVE_DIR, margin=16, preview_size=1024):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Hafif blur + sabit threshold: çizgiyi beyaz, arka plan siyah
    g = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(bw > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Kenarlardan güvenlik payı
    y1 = max(0, y1 - margin)
    x1 = max(0, x1 - margin)
    y2 = min(bw.shape[0]-1, y2 + margin)
    x2 = min(bw.shape[1]-1, x2 + margin)

    # Orijinal kırpım (native)
    crop = bw[y1:y2+1, x1:x2+1]

    # Karele (pad ekle)
    h, w = crop.shape
    side = max(h, w)
    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    square = cv2.copyMakeBorder(
        crop, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )

    square_native = square.copy()

    preview = cv2.resize(square_native, (preview_size, preview_size), interpolation=cv2.INTER_NEAREST)

    kernel = np.ones((3,3), np.uint8)
    thick = cv2.dilate(square_native, kernel, iterations=1)
    smooth = cv2.GaussianBlur(thick, (3,3), 0)
    model_28 = cv2.resize(smooth, (28, 28), interpolation=cv2.INTER_AREA)

    # Çoğu model siyah zemin üstünde beyaz karakter bekler -> invert
    model_28 = 255 - model_28

    base = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path_crop   = os.path.join(save_dir, f"{base}_crop.png")       # karelenmiş orijinal çözünürlük
    path_prev   = os.path.join(save_dir, f"{base}_preview.png")    # 1024x1024 (veya preview_size)
    path_model  = os.path.join(save_dir, f"{base}_28x28.png")      # model girişi

    cv2.imwrite(path_crop, square_native)
    cv2.imwrite(path_prev, preview)
    cv2.imwrite(path_model, model_28)

    return {"crop": path_crop, "preview": path_prev, "model": path_model, "side": int(side)}

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('camera did not open')
        return
        
    ok, frame = cap.read()
    if not ok:
        print('frame did not read')
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

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            index_tip_px = None
            pinch_now = False

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                idx_tip = hand.landmark[8]   # işaret parmağı ucu
                thb_tip = hand.landmark[4]   # başparmak ucu

                d = norm_dist(idx_tip, thb_tip)
                pinch_now = d < PINCH_THRESHOLD

                ix, iy = int(idx_tip.x * W), int(idx_tip.y * H)
                index_tip_px = (ix, iy)

                # İskeleti çizmek isterseniz:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            if index_tip_px is not None:
                # Basit smoothing
                if filtered_point is None:
                    filtered_point = np.array(index_tip_px, dtype=np.float32)
                else:
                    filtered_point = SMOOTHING * filtered_point + (1 - SMOOTHING) * np.array(index_tip_px, dtype=np.float32)

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

            # Önizleme
            blend = cv2.addWeighted(frame, 0.7, canvas, 0.9, 0)

            # UI metin ve pinch durumu
            cv2.putText(blend, "Pinch=draw. N=save, C=clear, Q=quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)
            status_color = (0, 255, 0) if drawing else (0, 0, 255)
            cv2.rectangle(blend, (10, 45), (30, 65), status_color, -1)

            cv2.imshow("Air-Draw Notepad (Step 1-3)", blend)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas[:] = 0
            elif key == ord('n'):
                path = preprocess_and_save(canvas)
                if path:
                    print(f"Saved: {path}")
                    canvas[:] = 0
                else:
                    print("Kaydedilecek çizim bulunamadı.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()