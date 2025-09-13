import cv2
from mtcnn import MTCNN

detector=MTCNN()

img=cv2.imread("D:\Projects\Surveillance\openCV\Detection_Models\Python Scripts\IMG_1641.jpg")
# h,w=img.shape[:2]
resized=cv2.resize(img,(1920,1080))

resized_rgb=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)

faces=detector.detect_faces(resized_rgb)

for face in faces:
    x,y,w,h=face['box']
    confidence=face['confidence']
    keypoints=face['keypoints']

    cv2.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)

    for point in keypoints.values():
        cv2.circle(resized,point,2,(0,255,0),-1)
    
    cv2.putText(resized,f"{confidence:.2f}",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

cv2.imshow("MTCNN_Detector",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()