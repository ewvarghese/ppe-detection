import cv2
import torch
from ultralytics import YOLO
import time

def main():

    print("CUDA Available:", torch.cuda.is_available())
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO("runs/detect/ppe_train_10_epochs_final/weights/best.pt")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam not accessible")
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, device=device, imgsz=512, conf=0.5)

        frame = results[0].plot()

        # FPS
        curr_time = time.time()
        fps = 1/(curr_time-prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.imshow("PPE Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()