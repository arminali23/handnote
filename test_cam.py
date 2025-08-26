import cv2

cap = cv2.VideoCapture(0)  # Gerekirse 1, 2 gibi başka indexleri deneyin
if not cap.isOpened():
    print("Kamera açılamadı.")
    raise SystemExit

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Camera Test - Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()