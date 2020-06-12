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

	# edges = cv2.Canny(dilation, 100, 200)
	out = np.dstack(( np.zeros_like(dilation) , np.zeros_like(dilation), np.zeros_like(dilation) )) * 255

	# Filter contours by area
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = [c for c in contours if cv2.contourArea(c) > 70]

	# cv2.drawContours(out, contours, -1, (0,255,0), 3)

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

		# cluster_means = []

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

		print(f'For {cluster_num} clusters, silhouette coefficient of {silhouette_coefficient_prev}. {silhouette_coefficient}')
		print(cluster_means)

		cluster_means = np.int32(cluster_means)

		for l in cluster_means:
			x1,y1,x2,y2 = l

			cv2.line(out, (x1,y1), (x2,y2), (0,0,255), 2, cv2.LINE_AA)

		# cv2.drawContours(out, [contours[i]], 0, (0,255,0), 3)


	return out


img = cv2.imread('lane_test.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

f, (ax1, ax2) = plt.subplots(2, 1)

ax1.imshow(thresh)
ax2.imshow(segmentation_v2(thresh))

plt.show()