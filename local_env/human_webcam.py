import cv2
from ultralytics import YOLO


model = YOLO("C:/Users/ASUS/runs/detect/train9\weights/best.pt")  

# Open the laptop webcam
cap = cv2.VideoCapture(2)  # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model.predict(source=frame, conf=0.4, stream=True)

    # Draw results on the frame
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLOv8 Human Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
