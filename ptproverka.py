import cv2

img = cv2.imread("phototest/t1.jpg")
img = cv2.resize(img, (224, 224))
cv2.imwrite("phototest/t1_repacked.png", img)