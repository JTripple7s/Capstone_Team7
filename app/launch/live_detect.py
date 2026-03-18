from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Open webcam (0 = default Mac camera)
cap = cv2.VideoCapture(0)

# Class names (make sure order matches your dataset)
CLASS_NAMES = ["ball", "goalpost"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Get annotated frame (with boxes drawn)
    annotated_frame = results[0].plot()

    # Extract detection info
    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Compute center
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Get class
            cls_id = int(box.cls[0])
            label = CLASS_NAMES[cls_id]

            # Print for navigation logic
            print(f"{label} -> center: ({int(x_center)}, {int(y_center)})")

            # Draw center point
            cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 255, 0), -1)

    # Show live window
    cv2.imshow("YOLO Live Detection", annotated_frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()