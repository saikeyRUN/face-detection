import cv2
import sys
import matplotlib.pyplot as plt

# Get user supplied valuesiflags = cv2.cv.CV_HAAR_SCALE_IMAGEcv2.startWindowThread()cv2.namedWindow("Faces found")cv2.imshow("Faces found", image)cv2.waitKey(0)cv2.destroyAllWindows()




cascPath = "haarcascade_frontalface_default.xml"

imagePath = sys.argv[1]
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


plt.imshow(image, cmap='gray')
plt.show()
