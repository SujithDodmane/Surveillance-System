# Thread----> For detection 
# Loop----> For reading and displaying the frames

import cv2
from facenet_pytorch import MTCNN
import torch
import threading
import time

#Detecting device if GPU available or not
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #torch.device('cuda') ---> operation happens on GPU  #torch.device('cpu') ---> operation happens on CPU

#Initializing the detector 
detector=MTCNN(keep_all=True, device=device)    

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
                boxes,probs,landmarks = detector.detect(frame_rgb,landmarks=True)
            except Exception as e:
                print("Detection error:", e)
                continue

            #without lock kooda kelsa agutte but one thread run agbeekadre innaondu thread i.e loop inda faces list corrupt aagbardu anta lock use madbeku
            with lock:
                faces=[]
                if boxes is not None:
                    for i,box in enumerate(boxes):
                        x1,y1,x2,y2=[int(a) for a in box]
                        confidence=probs[i]
                        keypoints=landmarks[i]
                        faces.append({'box': [x1,y1,x2,y2],'confidence': confidence,
                                      'keypoints': {'left_eye': keypoints[0],
                                                    'right_eye': keypoints[1],
                                                    'nose': keypoints[2],
                                                    'mouth_left': keypoints[3],
                                                    'outh_right': keypoints[4]
                                                    }})
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
                x1,y1,x2,y2=face['box']
                confidence=face['confidence']
                keypoints=face['keypoints']
                print(face)


                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

                for point in keypoints.values():
                    px=int(point[0])
                    py=int(point[1])
                    cv2.circle(frame,(px,py),2,(0,255,0),-1)
            
                cv2.putText(frame,f"Confidence:{confidence:.2f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,0),2)
    
        frame_count+=1
        elapsed_time=time.time()-start_time
        fps=frame_count/elapsed_time
        cv2.putText(frame,f"FPS:{fps:.2f}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)

        cv2.imshow("Survillance Security System(Face Detector)",frame)

        if cv2.waitKey(1) &0xFF==ord('q'):
            print("Closing the program")
            thread_stop_flag=True # Stops the detection as soon as reading stops
            latest_frame=None
            cv2.destroyAllWindows()
            break

    camera.release()

    
detection_thread=threading.Thread(target=face_detection,daemon=True)
read_display_thread=threading.Thread(target=read_display,daemon=True)


detection_thread.start()
read_display_thread.start()
read_display_thread.join()
detection_thread.join() #waits for the thread to finish the its blocks execution





