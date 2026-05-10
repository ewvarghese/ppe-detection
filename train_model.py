from ultralytics import YOLO
import torch
import os

def main():

    # ----------------------------
    # 1. Check GPU
    # ----------------------------
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("GPU not available. Using CPU.")
        device = "cpu"

    # ----------------------------
    # 2. Load YOLOv8 model
    # ----------------------------
    model = YOLO("yolov8n.pt")

    # ----------------------------
    # 3. Train Model (Stable Config)
    # ----------------------------
    results = model.train(
        data="PPE-detection-1/data.yaml",
        epochs=50,
        imgsz=512,        # Reduced for 4GB GPU stability
        batch=8,          # Safe for RTX 3050 4GB
        device=device,
        workers=0,        # IMPORTANT for Windows
        amp=True,
        name="ppe_train_10_epochs_final"
    )

    # ----------------------------
    # 4. Print Results
    # ----------------------------
    print("\n========== TRAINING COMPLETE ==========\n")

    save_dir = results.save_dir
    print("Training directory:", save_dir)

    best_model = os.path.join(save_dir, "weights", "best.pt")
    last_model = os.path.join(save_dir, "weights", "last.pt")

    print("Best model path:", best_model)
    print("Last model path:", last_model)

    # ----------------------------
    # 5. Validate Best Model
    # ----------------------------
    print("\n========== FINAL METRICS ==========\n")

    best = YOLO(best_model)
    metrics = best.val()

    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    print("\nModel ready for real-time detection.")


if __name__ == "__main__":
    main()