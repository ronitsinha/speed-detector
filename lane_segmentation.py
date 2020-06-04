import cv2
import numpy as np

from matplotlib import pyplot as plt
import scipy.signal as sp
import itertools
import operator

img = cv2.imread('lane_test.png')[:,0:300]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)


roi = thresh[550:600]

# for i in range(6):

# 	f, (ax1, ax2) = plt.subplots(1, 2)

# 	roi = img[i*100:(i*100)+100]
# 	hist = np.sum(roi, axis=0)


# 	ax1.imshow(roi)
# 	ax2.plot(hist)


# This algorithm SUCKS, read this paper instead!
# https://arxiv.org/pdf/1501.03124.pdf
def segmentation (binary, passes=5):
	curves_x = []
	curves_y = []
	peaks = []

	out = np.zeros_like(binary)

	height = binary.shape[0]//passes

	# TODO: make windows from the BOTTOM-UP!
	for i in range(passes):
		window = binary[ i*height : (i+1)*height ]

		hist = (np.sum(window, axis=0)).nonzero()[0]

		hsplit = np.split(hist,  np.where( np.diff(hist)!=1 )[0]+1 )


		nonzeros = np.transpose(window.nonzero())
		
		nonzeros = nonzeros [ nonzeros[:,1].argsort() ]

		nonzero_x = nonzeros[:,1]
		nonzero_y = nonzeros[:,0]

		# Split where not consecutive
		nonzero_x_grouped = np.split(nonzero_x,  np.where( np.diff(nonzero_x)>1 )[0]+1 )
		nonzero_y_grouped = np.split(nonzero_y,  np.where( np.diff(nonzero_x)>1 )[0]+1 )

		print(len(nonzero_x_grouped))

		try:
			for gy, gx in zip(nonzero_y_grouped[1],nonzero_x_grouped[1]):
				out[gy+i*height, gx] = 1
		except Exception as e:
			print('Error')
			continue

			

		# Now we have the pixels grouped into separate arrays (representing separate curves, hopefully)
		# if curves:

		# 	if len(nonzeros_x_grouped) != len(curves_x): 
		# 		# New lane or one lane disappeared
		# 		for j in range( len(nonzeros_x_grouped) ):
		# 			if j >= len(curves_x):
		# 				curves_x.append( nonzeros_x_grouped[j] )
		# 				curves_y.append( nonzeros_y_grouped[j] )
		# 			else:
		# 				curves_x[j] = np.concatenate( (curves_x[j], nonzeros_x_grouped[j]), axis=0)
		# 				curves_y[j] = np.concatenate( (curves_y[j], nonzeros_y_grouped[j]), axis=0)
		# 	else:
		# 		curves_x = np.concatenate( (curves_x,nonzeros_x_grouped), axis=1 )
		# 		curves_y = np.concatenate( (curves_y,nonzeros_y_grouped), axis=1 )

		# else:
		# 	# Initialize curves!
		# 	for group_x,group_y in zip(nonzeros_x_grouped, nonzeros_y_grouped):
		# 		curves_x.append(group_x)
		# 		curves_y.append(group_y)

	
	# TODO: polyfit stuff
	# fits = [ np.polyfit(group_x, group_y, 2)
	# 		for group_x, group_y in zip(curves_x, curves_y) ]

	return out		


f, (ax1, ax2) = plt.subplots(1, 2)

# hist = np.sum(roi, axis=0)

ax1.imshow(thresh)
ax2.imshow(segmentation(thresh,3))

plt.show()