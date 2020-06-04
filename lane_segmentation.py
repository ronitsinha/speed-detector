import cv2
import numpy as np

from matplotlib import pyplot as plt
import scipy.signal as sp
import itertools
import operator

img = cv2.imread('lane_test.png')[:,0:300]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)


roi = thresh[500:600]

for i in range(6):

	f, (ax1, ax2) = plt.subplots(1, 2)

	roi = img[i*100:(i*100)+100]
	hist = np.sum(roi, axis=0)


	ax1.imshow(roi)
	ax2.plot(hist)


# f, (ax1, ax2) = plt.subplots(1, 2)

# hist = np.sum(roi, axis=0)

# ax1.imshow(thresh)
# ax2.plot(hist)

plt.show()


def segmentation (binary, passes=5):
	curves = []
	peaks = []

	height = binary.shape[0]//passes

	for i in range(passes):
		window = binary[i*height:(i+1)*height]
		hist = np.sum(window, axis=0)

		nonzeros = np.transpose(window.nonzero())
		nonzero_x = window.nonzero()[0]

		# Split where not consecutive
		nonzeros_grouped = np.split(nonzeros,  np.where( np.diff(nonzero_x)!=1 )[0]+1 )
		
		# Now we have the pixels grouped into separate arrays (representing separate curves, hopefully)
		if curves:

			if len(nonzeros_grouped) != len(curves): 
				# New lane or one lane disappeared
				for j in range( len(nonzeros_grouped) ):
					if j >= len(curves):
						curves.append( nonzeros_grouped[j] )
					else:
						curves[j] = np.concatenate( (curves[j], nonzeros_grouped[j]), axis=0)
			else:
				curves = np.concatenate( (curves,nonzeros_grouped), axis=1 )

		else:
			# Initialize curves!
			for group in nonzeros_grouped:
				curves.append(group)

	
	# TODO: polyfit stuff
		

		


