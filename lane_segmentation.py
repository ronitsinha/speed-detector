import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

import sklearn.cluster # KMeans
import sklearn.metrics # silhouette_score

# https://www.ingentaconnect.com/contentone/ist/ei/2016/00002016/00000014/art00011?crawler=true
def segmentation_v2 (binary):

	kernel = np.ones((4,4), np.uint8)

	dilation = cv2.dilate(binary, kernel, iterations=1)

	out = np.dstack(( np.zeros_like(dilation) , np.zeros_like(dilation), np.zeros_like(dilation) )) * 255

	# Filter contours by area
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = [c for c in contours if cv2.contourArea(c) > 1000]

	groups = {}

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
			
			for j in range(len(contours)):
				if cv2.pointPolygonTest(contours[j], mid, True) >= 0:
					if j in groups:
						groups[j] = np.append(groups[j], [line], axis=0)
					else:
						groups[j] = np.array([line])
					break
	
	# For K-means
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

	for i, lines in groups.items():

		# K-means clustering
		silhouette_coefficient = 0.5
		silhouette_coefficient_prev = -1
		cluster_num = 1

		midpoints = np.array([ [(x1+x2)//2,(y1+y2)//2] for x1,y1,x2,y2 in lines ])
		midpoints = np.float32(midpoints)

		cluster_means = []

		while silhouette_coefficient > silhouette_coefficient_prev:

			if cluster_num >= len(lines):
				break

			cluster = sklearn.cluster.KMeans(n_clusters=cluster_num)	
			cluster_labels = cluster.fit_predict(midpoints)

			cluster_means = []
			for j in range(cluster_num):
				cluster_means.append( np.mean(lines[cluster_labels==j],axis=0) )

			if cluster_num > 1:
				silhouette_coefficient_prev = silhouette_coefficient
				silhouette_coefficient = sklearn.metrics.silhouette_score(midpoints, cluster_labels)

			cluster_num += 1

		cluster_num -= 1


		x_pos = np.array([[x1,x2] for x1,_,x2,_ in lines]).ravel()
		y_pos = np.array([[y1,y2] for _,y1,_,y2 in lines]).ravel()
		
		fit = np.polyfit(x_pos,y_pos,2)
		draw_x = np.linspace(np.min(x_pos), np.max(x_pos))
		draw_y = np.polyval(fit, draw_x)

		pts = (np.asarray([draw_x, draw_y]).T).astype(np.int32)

		color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
		cv2.polylines(out, [pts], False, color, 3)

		# for l in np.int32(lines):
		# 	x1,y1,x2,y2 = l

		# 	cv2.line(out, (x1,y1), (x2,y2), color, 2, cv2.LINE_AA)

	return out

if __name__ == '__main__':

	img = cv2.imread('lane_test2.png')

	roi = img[100:640]

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