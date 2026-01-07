import time
import cv2
import requests

VERIFY_URL = "http://localhost:8000/verify"
USER_ID = "rahum"
THRESHOLD = 0.35

def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try changing VideoCapture(0) to (1).")

    print("Press 'q' to quit.")
    last_sent = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # Show preview window
        cv2.imshow("Camera", frame)

        # Send a frame every 0.5 seconds
        now = time.time()
        if now - last_sent >= 0.5:
            last_sent = now

            # Encode frame as JPEG bytes
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                print("Failed to encode frame.")
                continue

            files = {"image": ("frame.jpg", buf.tobytes(), "image/jpeg")}
            params = {"user_id": USER_ID, "threshold": THRESHOLD}

            try:
                r = requests.post(VERIFY_URL, params=params, files=files, timeout=3)
                data = r.json()
                status = data.get("status")
                score = data.get("score")
                if status == "APPROVED":
                    print(f"APPROVED (score={score:.3f})")
                elif status == "DENIED":
                    print(f"DENIED (score={score:.3f})")
                else:
                    print(status)
            except Exception as e:
                print("Request failed:", e)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
