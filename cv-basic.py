import imutils
import cv2

# load the input image and show its dimensions, keeping in mind that
# images are represented as a multi-dimensional NumPy array with
# shape no. rows (height) x no. columns (width) x no. channels (depth)

image = cv2.imread("jp.png")
(h, w, d) = image.shape 
print("width={}, height={}, depth={}".format(w, h, d))
#dsiplaying image
cv2.imshow("Image", image)
cv2.waitKey(0)

# access the RGB pixel located at x=50, y=100, keepind in mind that
# OpenCV stores images in BGR order rather than RGB

(B, G, R) = image[100, 50] # height (y) comes before width (x)
print("R={}, G={}, B={}".format(R, G, B))

#Array slicing and cropping to extract Region of Interest (ROI)
# extract a 100x100 pixel square ROI (Region of Interest) from the input image starting at x=320,y=60 at ending at x=420,y=160
roi = image[60:160, 320:420]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

# Resizing Images
# resize the image to 200x200px, ignoring aspect ratio
#resized = cv2.resize(image, (200, 200))
#cv2.imshow("Fixed Resizing", resized)
#cv2.waitKey(0)

#resizing with aspect ratio, taking width as 300
r = 300.0 / w
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
# cv2.imshow("Aspect Ratio Resize", resized)
# cv2.waitKey(0)
# with imutils
resized = imutils.resize(image, width = 300)
cv2.imshow("Imutils Resize", resized)
cv2.waitKey(0)

# Rotating Images
# let's rotate an image 45 degrees clockwise 
# computing the image center, then constructing the rotation matrix,
# and then finally applying the affine warp
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotate = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated", rotate)
cv2.waitKey(0)

# imutils handles all the math for you
rotate = imutils.rotate_bound(image, -45)  # rotate_bound prevents cropping
cv2.imshow("Rotated", rotate)
cv2.waitKey(0)
# making sure the image isn't clipped
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)

# Smoothing
# apply a Gaussian blur with a 11x11 kernel to the image to smooth it,
# useful when reducing high frequency noise
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

# Drawing on an Image
# draw a 2px thick red rectangle surrounding the face, then draw a
output = image.copy()
cv2.rectangle(output, (50, 30), (200, 130), (0, 0, 255), 2)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)  

# draw a 5px thick red line from x=60,y=20 to x=400,y=200
output = image.copy()
cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
cv2.imshow("Line", output)
cv2.waitKey(0)

# draw green text on the image
output = image.copy()
cv2.putText(output, "OpenCV + Jurassic Park!!!", (10, 25), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Text", output)
cv2.waitKey(0)


