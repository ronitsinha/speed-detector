import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

import sklearn.cluster # KMeans
import sklearn.metrics # silhouette_score

dist2 = lambda a,b: (a[0]-b[0])**2 + (a[1]-b[1])**2

# https://www.ingentaconnect.com/contentone/ist/ei/2016/00002016/00000014/art00011?crawler=true
def segmentation_v2 (binary):

	kernel = np.ones((4,4), np.uint8)
	dilation = cv2.dilate(binary, kernel, iterations=1)

	out = np.dstack(( np.zeros_like(dilation) , np.zeros_like(dilation), np.zeros_like(dilation) )) * 255

	# Filter contours by area
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = [c for c in contours if cv2.contourArea(c) > 350]
	sorted_ctrs = sorted(contours, key=lambda c: cv2.boundingRect(c)[0] )
	
	groups = [None]*len(sorted_ctrs)

	lane_x = []
	lane_y = []

	linesP = cv2.HoughLinesP(dilation, 1, np.pi / 180, 20, None, 2, 20)
	if linesP is not None:
		for i in range(0, len(linesP)):
			x1,y1,x2,y2 = linesP[i][0]			

			line = []

			if y1 > y2:
				line = [x1,y1,x2,y2]
			else:
				line = [x2,y2,x1,y1]

			mid = ( (x1+x2)//2, (y1+y2)//2 )
			
			for j in range(len(sorted_ctrs)):
				if cv2.pointPolygonTest(sorted_ctrs[j], mid, True) >= 0:
					if groups[j] is not None:
						groups[j] = np.append(groups[j], [line], axis=0)
					else:
						groups[j] = np.array([line])
					break
	
	for lines in groups:

		x_pos = np.array([[x1,x2] for x1,_,x2,_ in lines]).ravel()
		y_pos = np.array([[y1,y2] for _,y1,_,y2 in lines]).ravel()
		
		fit = np.polyfit(x_pos,y_pos,2)
		draw_x = np.linspace(np.min(x_pos), np.max(x_pos))
		draw_y = np.polyval(fit, draw_x)

		lane_x = np.append(lane_x, draw_x)
		lane_y = np.append(lane_y, draw_y)

		pts = (np.asarray([draw_x, draw_y]).T).astype(np.int32)

		# color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
		# cv2.polylines(out, [pts], False, color, 3)
	
	pts = np.int32( np.column_stack((lane_x, lane_y)) )

	# increment by 50 b/c each polyfit has 50 sample points
	for i in range(0,len(pts)-50,50):
		lane = np.concatenate( (pts[i:i+50], np.flip(pts[i+50:i+100], axis=0) ) )
		color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
		cv2.fillPoly(out, np.array([lane]), color)

	# https://stackoverflow.com/questions/58377015/counterclockwise-sorting-of-x-y-data

	# cv2.polylines(out, np.array([pts]), False, (0,200,255), 1)
	# cv2.fillPoly(out, np.array([pts]), (0,200,255))
	cv2.drawContours(out, sorted_ctrs, -1, (0,255,0))

	return out

if __name__ == '__main__':

	img = cv2.imread('lane_test.png')

	roi = img[50:640,50:480]

	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

	lower_yellow = np.array([0, 27, 161])
	upper_yellow = np.array([40, 255, 255])

	# Breakthrough: dashed white lines don't matter! they don't separate opposite traffic!
	# More info here: https://www.123driving.com/dmv/drivers-handbook-pavement-markings
	_, mask_white = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
	mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
	color_mask = cv2.bitwise_or(mask_yellow, mask_white)

	f, (ax1, ax2) = plt.subplots(2, 1)
	ax1.imshow(color_mask)
	ax2.imshow(segmentation_v2(color_mask))

	plt.show()