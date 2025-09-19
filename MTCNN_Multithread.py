# Thread----> For detection 
# Loop----> For reading and displaying the frames

import cv2
from mtcnn import MTCNN
import threading
import time

#Initializing the detector 
detector=MTCNN()

#Global variables that can be used by both thread and loop 
faces=[]
latest_frame=None
thread_stop_flag=False
lock=threading.Lock()


def face_detection():
    
    global faces,latest_frame,thread_stop_flag # To specify these are global variables

    while not thread_stop_flag:
        if latest_frame is not None:
            print("Running detection....")
            frame_copy = latest_frame.copy()
            frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

            try:
                results = detector.detect_faces(frame_rgb)
            except Exception as e:
                print("Detection error:", e)
                continue

            #without lock kooda kelsa agutte but one thread run agbeekadre innaondu thread i.e loop inda faces list corrupt aagbardu anta lock use madbeku
            with lock:
                faces=results
    print("Thread exitting....")






frame_count=0
start_time=time.time()

def read_display():
    global faces,latest_frame,frame_count,start_time,thread_stop_flag
    camera=cv2.VideoCapture(0)

    while True:
        success,frame=camera.read()
        if not success:
            break

        frame=cv2.resize(frame,(1440,820))

        latest_frame=frame


        with lock:
            for face in faces:
                x,y,w,h=face['box']
                confidence=face['confidence']
                keypoints=face['keypoints']
                print(face)


                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                for point in keypoints.values():
                    cv2.circle(frame,point,2,(0,255,0),-1)
            
                cv2.putText(frame,f"Confidence:{confidence:.2f}",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,0),2)
    
        frame_count+=1
        elapsed_time=time.time()-start_time
        fps=frame_count/elapsed_time
        cv2.putText(frame,f"FPS:{fps:.2f}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)

        cv2.imshow("Survillance Security System(Face Detector)",frame)

        if cv2.waitKey(1) &0xFF==ord('q'):
            print("Closing the program")
            thread_stop_flag=True # Stops the detection as soon as reading stops
            latest_frame=None
            break

    camera.release()

    
detection_thread=threading.Thread(target=face_detection,daemon=True)
read_display_thread=threading.Thread(target=read_display,daemon=True)


detection_thread.start()
read_display_thread.start()
read_display_thread.join()
detection_thread.join() #waits for the thread to finish the its blocks execution


cv2.destroyAllWindows()


