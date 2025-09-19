import cv2
import torch
from facenet_pytorch import MTCNN
import time

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize MTCNN with GPU
mtcnn = MTCNN(keep_all=True, device=device)

# Open RTSP camera
camera = cv2.VideoCapture("rtsp://admin:dodmane%407854@192.168.0.110:554/cam/realmonitor?channel=2&subtype=0")

prev_time = 0

while True:
    success, frame = camera.read()
    if not success:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (1440, 820))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces (returns boxes, probabilities, and landmarks)
    boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)

    if boxes is not None:
        for i, box in enumerate(boxes):
            print(box)
            x1, y1, x2, y2 = [int(b) for b in box]
            confidence = probs[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw keypoints
            for point in landmarks[i]:
                cv2.circle(frame, tuple(point.astype(int)), 2, (0, 255, 0), -1)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Smart Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
