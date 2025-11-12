# detection.py
# Thread----> For detection 
# Uses state.latest_frame and writes to state.faces

import cv2
from facenet_pytorch import MTCNN

import state

def face_detection(detector):
    global_detector = detector
    while not state.thread_stop_flag:
        if state.latest_frame is not None:
            try:
                frame_copy = state.latest_frame.copy()
            except Exception:
                # frame may be invalid temporarily
                continue

            frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

            try:
                boxes, probs, landmarks = global_detector.detect(frame_rgb, landmarks=True)
            except Exception as e:
                print("Detection error:", e)
                continue

            #without lock kooda kelsa agutte but one thread run agbeekadre innaondu thread i.e loop inda faces list corrupt aagbardu anta lock use madbeku
            with state.lock:
                # use clear() to keep same list object
                state.faces.clear()
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = [int(a) for a in box]
                        confidence = probs[i]
                        keypoints = landmarks[i]
                        state.faces.append({
                            'box': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'keypoints': {
                                'left_eye': keypoints[0],
                                'right_eye': keypoints[1],
                                'nose': keypoints[2],
                                'mouth_left': keypoints[3],
                                'mouth_right': keypoints[4]
                            }
                        })
    print("Thread exitting....")
    cv2.destroyAllWindows()
