from ultralytics import YOLO
import cv2
import os

model = YOLO("models/yolo11n.pt")

def run_detection(image_path: str):
    # Run inference
    results = model(image_path)

    # Read original image for drawing
    img = cv2.imread(image_path)

    detections = []

    for result in results:
        for box in result.boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Label name, confidence
            cls_id = int(box.cls[0])
            label = result.names[cls_id]
            conf = float(box.conf[0])

            # Save detection info
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

            # Draw box + label on image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img,
                        f"{label} {conf:.2f}",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

    # Save annotated output
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/annotated_bol.jpg"
    cv2.imwrite(out_path, img)

    print(f"Saved annotated image → {out_path}")
    return detections, out_path


# ---------- Example Run ----------
if __name__ == "__main__":
    detections, path = run_detection("inputs/test1.png")
    print("Detections:", detections)
