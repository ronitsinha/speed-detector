import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lane_test.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)


roi = thresh

# for i in range(6):

# 	f, (ax1, ax2) = plt.subplots(1, 2)

# 	roi = img[i*100:(i*100)+100]
# 	hist = np.sum(roi, axis=0)


# 	ax1.imshow(roi)
# 	ax2.plot(hist)


# This algorithm SUCKS, read this paper instead!
# https://arxiv.org/pdf/1501.03124.pdf

# TODO: use distance to decide if a curve can be added to existing curve + bottom-up search
def segmentation (binary, passes=5, diff_threshold=0.2):
	curves_x = np.array([])
	curves_y = np.array([])
	discontinuity = np.array([])

	out = np.zeros_like(binary)

	img_height = binary.shape[0]
	window_height = img_height//passes

	for i in range(passes):
		# Bottom-up search
		window = binary[ img_height-(i+1)*window_height : img_height-i*window_height ]

		nonzeros = np.transpose(window.nonzero())
		nonzeros = nonzeros [ nonzeros[:,1].argsort() ]

		nonzero_x = nonzeros[:,1]
		nonzero_y = nonzeros[:,0]

		# Split where not consecutive
		nonzero_x_grouped = np.array( np.split(nonzero_x,  np.where( np.diff(nonzero_x)>1 )[0]+1 ) )
		nonzero_y_grouped = np.array( np.split(nonzero_y,  np.where( np.diff(nonzero_x)>1 )[0]+1 ) ) - (i+1)*window_height


		# try:
		# 	for gy, gx in zip(nonzero_y_grouped[0],nonzero_x_grouped[0]):
		# 		out[gy-(i+1)*window_height, gx] = 1
		# except Exception as e:
		# 	print('Error')
		# 	continue

			
		if len(curves_x) <= 0:
			curves_x = nonzero_x_grouped
			curves_y = nonzero_y_grouped
			discontinuity = np.ones_like(curves_x)
			continue

		# for some reason, curves is not considered a np.array, just a normal list! FIX THIS

		prev_x = np.array([ c[-1] for c in curves_x ])

		print(nonzero_x_grouped)


		# Temporary thing
		for j in range(len(prev_x)):
			thresh = discontinuity[j]*diff_threshold
			avg_x = np.array([ np.average(group) for group in nonzero_x_grouped ])
			x = prev_x[j]

			dist = np.array( np.absolute(avg_x-x)/x )
			
			if len(dist[dist<thresh]) <= 0 or len(nonzero_x_grouped) <= 0:
				discontinuity[j] += 1
				continue

			discontinuity[j] = 1
			closest = np.argmin(dist[dist<thresh])


			curves_x[j] = np.append(curves_x[j], nonzero_x_grouped[closest])
			curves_y[j] = np.append(curves_y[j], nonzero_y_grouped[closest])

			# This is really sloppy and causes index errors
			nonzero_x_grouped = np.delete(nonzero_x_grouped, closest)
			nonzero_y_grouped = np.delete(nonzero_y_grouped, closest)


		# New groups -- concat/append!




		# Now we have the pixels grouped into separate arrays (representing separate curves, hopefully)
		# if len(curves_x) > 0:

		# 	if len(nonzero_x_grouped) != len(curves_x): 
		# 		# New lane or one lane disappeared
		# 		for j in range( len(nonzero_x_grouped) ):
		# 			if j >= len(curves_x):
		# 				curves_x = np.append( curves_x, nonzero_x_grouped[j] )
		# 				curves_y = np.append( curves_x, nonzero_y_grouped[j] )
		# 			else:
		# 				curves_x[j] = np.concatenate( (curves_x[j], nonzero_x_grouped[j]), axis=0)
		# 				curves_y[j] = np.concatenate( (curves_y[j], nonzero_y_grouped[j]), axis=0)
		# 	else:
		# 		curves_x = np.concatenate( (curves_x,nonzero_x_grouped), axis=1 )
		# 		curves_y = np.concatenate( (curves_y,nonzero_y_grouped), axis=1 )

		# else:
		# 	# Initialize curves!
		# 	for group_x,group_y in zip(nonzero_x_grouped, nonzero_y_grouped):
		# 		curves_x = np.append(curves_x, group_x)
		# 		curves_y = np.append(curves_y, group_y)

	
	print(curves_x)

	# for gy, gx in zip(curves_y, curves_x):
	# 	out[gy, gx] = 1

	out[curves_y[1], curves_x[1]] = 1

	# TODO: polyfit stuff
	# fits = [ np.polyfit(group_x, group_y, 2)
	# 		for group_x, group_y in zip(curves_x, curves_y) ]

	return out		


f, (ax1, ax2) = plt.subplots(2, 1)

# hist = np.sum(roi, axis=0)

ax1.imshow(thresh)
ax2.imshow(segmentation(thresh,50))

plt.show()