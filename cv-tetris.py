import cv2
import imutils
import argparse 

# construct the argument parser and parse the arguments#
# args["image"] in the script, we’re referring to the path to the input image.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)

# convert image into grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

# edge detection
edged = cv2.Canny(gray, 30, 150)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

# thresholding
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

# detecting and drawing contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()
for c in cnts:
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3) # purple outline
    cv2.imshow("Contours", output)
    cv2.waitKey(0)
text = "{} objects found".format(len(cnts))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)

# erosions and dilations
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations = 5)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

# masking and bitwise operations
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Masked", output)
cv2.waitKey(0)


