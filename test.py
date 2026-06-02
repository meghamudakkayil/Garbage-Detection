import cv2
import imutils
image = cv2.imread("jp.png")
(h, w, d) = image.shape
roi = image[60:160, 320:420]
cv2.imshow("ROI", roi)
cv2.waitKey(0)
 