import cv2
import face_recognition

# Load image with OpenCV (BGR)
image = cv2.imread("/mnt/d/Projects/Surveillance-System/Face_recognition/img_faces.jpg")


# Resize the image to 50% of its original size
scale_percent = 25   # reduce to 25 if still too heavy
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Convert to RGB (face_recognition expects RGB)
rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# Detect faces
face_locations = face_recognition.face_locations(
    rgb_image,
    number_of_times_to_upsample=1,
    model="hog"   # use "cnn" if GPU available, "hog" is faster on CPU
)

print("Found faces:", len(face_locations))

# Draw rectangles around faces
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(resized_image, (left, top), (right, bottom), (0, 255, 0), 2)

# Show image
cv2.imshow("Detected Faces", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
