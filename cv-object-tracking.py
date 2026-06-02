from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import sys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

try:
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
# these are the various tracking algorithms 
except AttributeError:
	# Fallback for newer OpenCV versions where trackers moved to cv2.legacy
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.legacy.TrackerCSRT_create,
		"kcf": cv2.legacy.TrackerKCF_create,
		"boosting": cv2.legacy.TrackerBoosting_create,
		"mil": cv2.legacy.TrackerMIL_create,
		"tld": cv2.legacy.TrackerTLD_create,
		"medianflow": cv2.legacy.TrackerMedianFlow_create,
		"mosse": cv2.legacy.TrackerMOSSE_create
	}
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

initBB = None # coordinates of bounding box

if not args.get("video", False):
    print("[INFO] starting live webcam stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
	print(f"[INFO] loading video file: {args['video']}")
	vs = cv2.VideoCapture(args["video"])

# initialize the FPS (frames per second) throughput estimator
fps = None
# loop over frames from the video stream
while True:
	# grab the current frame
	frame = vs.read()
	# handle if we are using a VideoStream or VideoCapture object
	frame = frame[1] if args.get("video", False) else frame
	
	# check to see if we have reached the end of the stream
	if frame is None:
		break
		
	# resize the frame and grab the frame dimensions
	frame = imutils.resize(frame, width=500) 
	(H, W) = frame.shape[:2]
	
    # check to see if we are currently tracking an object
	if initBB is not None:
		# grab the new bounding box coordinates of the object
		(success, box) = tracker.update(frame)
		# check to see if the tracking was a success
		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h),
				(0, 255, 0), 2)
		
		# update the FPS counter
		fps.update()
		fps.stop() 
		
		# initialize the set of information we'll be displaying on the frame
		info = [
			("Tracker", args["tracker"]),
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps.fps())),
		]
		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			
    # show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	# If the 's' key is pressed, select a bounding box to track
	if key == ord("s"):
		initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
		tracker.init(frame, initBB)
		fps = FPS().start()
		
    # if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

# Clean up resources safely
if not args.get("video", False):
	vs.stop()
else:
	vs.release()

cv2.destroyAllWindows()

# more tools:
# background subtraction 
backSub = cv2.createBackgroundSubtractorMOG2()
fgMask = backSub.apply(frame)
cv2.imshow("Mask", fgMask)
# colour detection : useful for isolating specific waste objects
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower = (29, 86, 6)
upper = (64, 255, 255)
mask = cv2.inRange(hsv, lower, upper)