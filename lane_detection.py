import time

import cv2
import numpy as np

url = 'http://wzmedia.dot.ca.gov:1935/D3/80_donner_lake.stream/index.m3u8'

lower_yellow = np.array([20, 38, 153], dtype='uint8')
upper_yellow = np.array([30, 255, 255], dtype='uint8')

kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)

cv2.namedWindow('Final')
cv2.createTrackbar('Threshold Min', 'Final', 0, 255, lambda _: None)
cv2.setTrackbarPos('Threshold Min', 'Final', 191)

cv2.createTrackbar('Threshold Max', 'Final', 0, 255, lambda _: None)
cv2.setTrackbarPos('Threshold Max', 'Final', 255)

cv2.createTrackbar('Low H', 'Final', 0, 180, lambda _: None)
cv2.createTrackbar('Low S', 'Final', 0, 255, lambda _: None)
cv2.createTrackbar('Low V', 'Final', 0, 255, lambda _: None)

cv2.createTrackbar('High H', 'Final', 0, 180, lambda _: None)
cv2.createTrackbar('High S', 'Final', 0, 255, lambda _: None)
cv2.createTrackbar('High V', 'Final', 0, 255, lambda _: None)

# Testing out some perspective stuff
whitmore_grade_pts = np.float32([ [111,94], [230,94], [336,480], [640,429] ])
donner_lake_pts = np.float32([ [270,180], [406,180], [284,480],[640,367] ])
screen_pts = np.float32([ [0,0], [400,0], [0,600], [400,600] ])

matrix = cv2.getPerspectiveTransform(donner_lake_pts, screen_pts)

cap = cv2.VideoCapture(url)

while True:
	lower_yellow = np.array([cv2.getTrackbarPos('Low H', 'Final'), cv2.getTrackbarPos('Low S', 'Final'), cv2.getTrackbarPos('Low V', 'Final')], dtype ='uint8')
	upper_yellow = np.array([cv2.getTrackbarPos('High H', 'Final'), cv2.getTrackbarPos('High S', 'Final'), cv2.getTrackbarPos('High V', 'Final')], dtype ='uint8')

	ret, raw = cap.read()

	if not ret:
		print('Couldn\'t get new frame!')
		break

	frame = raw.copy()

	roi = frame[ 180:480, 270:640 ]

	# TODO: In the actual program, use THIS instead of cropped regions
	warped = cv2.warpPerspective(frame, matrix, (400,600))

	gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

	fg_mask = bg_subtractor.apply(gray)

	# Histogram equilization
	# https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html

	# These do not work when its darker out! ARGH!
	mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
	_, mask_white = cv2.threshold(gray, cv2.getTrackbarPos('Threshold Min', 'Final'), cv2.getTrackbarPos('Threshold Max', 'Final'), cv2.THRESH_BINARY)
	lane_mask = cv2.bitwise_or(mask_yellow, mask_white)

	# TODO: I don't know if the subtract is the right operation. Look into this more!!
	subtract = cv2.subtract(lane_mask, fg_mask)
	blur = cv2.GaussianBlur(subtract, (5,5), 0)


	# Edge detection
	edges = cv2.Canny(subtract, 100, 200)

	linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, None, 10, 50)

	if linesP is not None:
		for i in range(0, len(linesP)):
			l = linesP[i][0]
			cv2.line(warped, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

	# TODO: Read this: https://www.hackster.io/kemfic/curved-lane-detection-34f771
	cv2.imwrite('lane_test.png', subtract)

	cv2.imshow('Source', frame)
	cv2.imshow('Final', subtract)
	cv2.imshow('Warped', warped)




	if cv2.waitKey(1) & 0xFF == 27:
		print('Exiting...')
		break

	time.sleep(1/30)

cap.release()
cv2.destroyAllWindows()