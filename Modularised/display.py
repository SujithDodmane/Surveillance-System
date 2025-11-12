# display.py
# Loop----> For reading and displaying the frames
# Handles main camera reading, drawing boxes, displaying FPS, and key bindings (E, S, Q)

import cv2
import time
import state
from embeddings_manager import register_embedding, save_embeddings

def read_display(recognizer, device):
    camera = cv2.VideoCapture(0)
    frame_count = 0
    start_time = time.time()

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.resize(frame, (1440, 820))
        # update the shared latest_frame
        state.latest_frame = frame.copy()

        #without lock kooda kelsa agutte but one thread run agbeekadre innaondu thread i.e loop inda faces list corrupt aagbardu anta lock use madbeku
        with state.lock:
            for i, face in enumerate(state.faces):
                x1, y1, x2, y2 = face['box']
                confidence = face['confidence']
                keypoints = face['keypoints']

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                for point in keypoints.values():
                    px, py = int(point[0]), int(point[1])
                    cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

                cv2.putText(frame, f"{i+1}.Confidence:{confidence:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 0), 2)

                if i in state.face_results:
                    print(f"{state.face_results[i]['similarity']}  {state.face_results[i]['name']}")
                    cv2.putText(frame, f"{i+1}.S:{state.face_results[i]['similarity']:.2f}", (x1, y1 - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
                    if state.face_results[i]['similarity'] > 0.55:
                        cv2.putText(frame, f"{i+1}.Name:{state.face_results[i]['name']}", (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
        cv2.putText(frame, f"FPS:{fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)

        cv2.imshow("Survillance Security System(Face Detector)", frame)

        key = cv2.waitKey(1) & 0xFF
        # set shared key so other threads can read it (Face_Recognition checks state.key)
        state.key = key

        # Key bindings (Keyboard actions)
        if key == ord('e'):
            #only one face should be there while registering the embeddings
            if len(state.faces_img_rgb) == 1:
                register_embedding(state.faces_img_rgb[0], "Sujith", recognizer, device, state.known_face_embeddings)
                print("Face_Embedding_Function called")
            else:
                print("⚠️ Make sure only one face is visible while pressing E")

        elif key == ord('s'):
            save_embeddings(state.known_face_embeddings, "face_embeddings_3.pkl")

        elif key == ord('q'):
            print("Closing the program")
            state.thread_stop_flag = True # Stops the detection as soon as reading stops
            state.latest_frame = None
            cv2.destroyAllWindows()
            break

    camera.release()
