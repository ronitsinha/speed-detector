import time

import cv2
import numpy as np

url = 'http://wzmedia.dot.ca.gov:1935/D3/80_whitmore_grade.stream/index.m3u8'

lower_yellow = np.array([20, 38, 153], dtype='uint8')
upper_yellow = np.array([30, 255, 255], dtype='uint8')

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)

cv2.namedWindow('Final')
cv2.createTrackbar('Threshold', 'Final', 0, 255, lambda _: None)
cv2.setTrackbarPos('Threshold', 'Final', 200)

# cv2.createTrackbar('Low H', 'Final', 0, 180, lambda _: None)
# cv2.createTrackbar('Low S', 'Final', 0, 255, lambda _: None)
# cv2.createTrackbar('Low V', 'Final', 0, 255, lambda _: None)

# cv2.createTrackbar('High H', 'Final', 0, 180, lambda _: None)
# cv2.createTrackbar('High S', 'Final', 0, 255, lambda _: None)
# cv2.createTrackbar('High V', 'Final', 0, 255, lambda _: None)

cap = cv2.VideoCapture(url)

while True:
	# lower_yellow = np.array([cv2.getTrackbarPos('Low H', 'Final'), cv2.getTrackbarPos('Low S', 'Final'), cv2.getTrackbarPos('Low V', 'Final')], dtype ='uint8')
	# upper_yellow = np.array([cv2.getTrackbarPos('High H', 'Final'), cv2.getTrackbarPos('High S', 'Final'), cv2.getTrackbarPos('High V', 'Final')], dtype ='uint8')

	ret, raw = cap.read()

	if not ret:
		print('Couldn\'t get new frame!')
		break

	frame = raw.copy()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	fg_mask = bg_subtractor.apply(gray)

	# These do not work when its darker out! ARGH!
	mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
	_, mask_white = cv2.threshold(gray, cv2.getTrackbarPos('Threshold', 'Final'), 255, cv2.THRESH_BINARY)
	lane_mask = cv2.bitwise_or(mask_yellow, mask_white)

	# TODO: I don't know if the subtract is the right operation. Look into this more!!
	subtract = cv2.subtract(lane_mask, fg_mask)

	# TODO: Maybe use opening instead?
	# blur = cv2.GaussianBlur(subtract, (3,3), 0)
	opening = cv2.morphologyEx(subtract, cv2.MORPH_OPEN, kernel_open)


	cv2.imshow('Source', frame)
	cv2.imshow('Color Mask', lane_mask)
	cv2.imshow('Background Subraction', fg_mask)
	cv2.imshow('Final', opening)
	cv2.imshow('Subtract', subtract)

	if cv2.waitKey(1) & 0xFF == 27:
		print('Exiting...')
		break

	time.sleep(1/30)

cap.release()
cv2.destroyAllWindows()