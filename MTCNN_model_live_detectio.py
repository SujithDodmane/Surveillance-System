import cv2
from mtcnn import MTCNN

detector=MTCNN()

camera=cv2.VideoCapture(0)

while True:
    success,frame_1=camera.read()
    if not success:
        break
    #frame=cv2.resize(frame_1,(1440,820))
    frame=frame_1
    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    faces=detector.detect_faces(frame_rgb)
    print(faces)
    for face in faces:
        x,y,w,h=face["box"]
        confidence=face["confidence"]
        keypoints=face["keypoints"]
        #print(keypoints)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        for point in keypoints.values():
            cv2.circle(frame,point,2,(0,255,0),-1)
        
        cv2.putText(frame,f"Confidence{confidence:.2f}",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    cv2.imshow("Smart Face Detector",frame)

    if cv2.waitKey(1) &0xFF==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
    
