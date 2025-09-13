import cv2
from mtcnn import MTCNN
import threading
import multiprocessing as mp
import time

def capture_thread(frame_buffer, stop_event):
    cap = cv2.VideoCapture(0)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame_buffer['frame'] = frame
    cap.release()

def display_thread(frame_buffer, faces_buffer, stop_event):
    while not stop_event.is_set():
        frame = frame_buffer.get('frame', None)
        if frame is None:
            continue
        # Draw detections
        faces = faces_buffer.get('faces', [])
        for face in faces:
            x, y, w, h = face["box"]
            keypoints = face["keypoints"]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            for point in keypoints.values():
                cv2.circle(frame, point, 2, (0,255,0), -1)
        cv2.imshow("Face Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    cv2.destroyAllWindows()

def detection_process(frame_queue, faces_queue, stop_event):
    detector = MTCNN()
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(frame_rgb)
            faces_queue.put(faces)

if __name__ == "__main__":
    manager = mp.Manager()
    frame_buffer = manager.dict()
    faces_buffer = manager.dict()
    frame_queue = mp.Queue(maxsize=1)
    faces_queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()

    # Start threads
    t1 = threading.Thread(target=capture_thread, args=(frame_buffer, stop_event))
    t2 = threading.Thread(target=display_thread, args=(frame_buffer, faces_buffer, stop_event))
    t1.start()
    t2.start()

    # Start detection process
    p = mp.Process(target=detection_process, args=(frame_queue, faces_queue, stop_event))
    p.start()

    try:
        while not stop_event.is_set():
            frame = frame_buffer.get('frame', None)
            if frame is not None and not frame_queue.full():
                frame_queue.put(frame)
            # Get latest detection results
            while not faces_queue.empty():
                faces_buffer['faces'] = faces_queue.get()
            time.sleep(0.001)
    except KeyboardInterrupt:
        stop_event.set()

    t1.join()
    t2.join()
    p.terminate()
    p.join()
