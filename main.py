import cv2
import numpy as np
import time
import copy
import os
import glob
import multiprocessing as mpr
from datetime import datetime

from kalman_filter import KalmanFilter
from tracker import Tracker



if __name__ == '__main__':
	# The one I first used for testing; after staring at it so much, I've grown attached to this road :3
	the_og_base_url = 'http://wzmedia.dot.ca.gov:1935/D3/89_rampart.stream/'

	BASE_URL = 'http://wzmedia.dot.ca.gov:1935/D3/80_whitmore_grade.stream/'
	FPS = 30
	'''
		Distance to line in road: ~0.025 miles
	'''
	ROAD_DIST_MILES = 0.025

	'''
		Speed limit of urban freeways in California (50-65 MPH)
	'''
	HIGHWAY_SPEED_LIMIT = 65

	# Initial background subtractor and text font
	fgbg = cv2.createBackgroundSubtractorMOG2()
	font = cv2.FONT_HERSHEY_PLAIN

	centers = [] 

	# y-cooridinate for speed detection line
	Y_THRESH = 240

	blob_min_width_far = 6
	blob_min_height_far = 6

	blob_min_width_near = 18
	blob_min_height_near = 18

	frame_start_time = None

	# Create object tracker
	tracker = Tracker(80, 3, 2, 1)

	# Capture livestream
	cap = cv2.VideoCapture (BASE_URL + 'playlist.m3u8')

	while True:
		centers = []
		frame_start_time = datetime.utcnow()
		ret, frame = cap.read()

		orig_frame = copy.copy(frame)

		#  Draw line used for speed detection
		cv2.line(frame,(0, Y_THRESH),(640, Y_THRESH),(255,0,0),2)


		# Convert frame to grayscale and perform background subtraction
		gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
		fgmask = fgbg.apply (gray)

		# Perform some Morphological operations to remove noise
		kernel = np.ones((4,4),np.uint8)
		kernel_dilate = np.ones((5,5),np.uint8)
		opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		dilation = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel_dilate)

		_, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Find centers of all detected objects
		for cnt in contours:
			x, y, w, h = cv2.boundingRect(cnt)

			if y > Y_THRESH:
				if w >= blob_min_width_near and h >= blob_min_height_near:
					center = np.array ([[x+w/2], [y+h/2]])
					centers.append(np.round(center))

					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
			else:
				if w >= blob_min_width_far and h >= blob_min_height_far:
					center = np.array ([[x+w/2], [y+h/2]])
					centers.append(np.round(center))

					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		if centers:
			tracker.update(centers)

			for vehicle in tracker.tracks:
				if len(vehicle.trace) > 1:
					for j in range(len(vehicle.trace)-1):
                        # Draw trace line
						x1 = vehicle.trace[j][0][0]
						y1 = vehicle.trace[j][1][0]
						x2 = vehicle.trace[j+1][0][0]
						y2 = vehicle.trace[j+1][1][0]

						cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

					try:
						'''
							TODO: account for load lag
						'''

						trace_i = len(vehicle.trace) - 1

						trace_x = vehicle.trace[trace_i][0][0]
						trace_y = vehicle.trace[trace_i][1][0]

						# Check if tracked object has reached the speed detection line
						if trace_y <= Y_THRESH + 5 and trace_y >= Y_THRESH - 5 and not vehicle.passed:
							cv2.putText(frame, 'I PASSED!', (int(trace_x), int(trace_y)), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
							vehicle.passed = True

							load_lag = (datetime.utcnow() - frame_start_time).total_seconds()

							time_dur = (datetime.utcnow() - vehicle.start_time).total_seconds() - load_lag
							time_dur /= 60
							time_dur /= 60

							
							vehicle.mph = ROAD_DIST_MILES / time_dur

							# If calculated speed exceeds speed limit, save an image of speeding car
							if vehicle.mph > HIGHWAY_SPEED_LIMIT:
								print ('UH OH, SPEEDING!')
								cv2.circle(orig_frame, (int(trace_x), int(trace_y)), 20, (0, 0, 255), 2)
								cv2.putText(orig_frame, 'MPH: %s' % int(vehicle.mph), (int(trace_x), int(trace_y)), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
								cv2.imwrite('speeding_%s.png' % vehicle.track_id, orig_frame)
								print ('FILE SAVED!')

					
						if vehicle.passed:
							# Display speed if available
							cv2.putText(frame, 'MPH: %s' % int(vehicle.mph), (int(trace_x), int(trace_y)), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
						else:
							# Otherwise, just show tracking id
							cv2.putText(frame, 'ID: '+ str(vehicle.track_id), (int(trace_x), int(trace_y)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
					except:
						pass


		# Display all images
		cv2.imshow ('original', frame)
		cv2.imshow ('opening/dilation', dilation)
		cv2.imshow ('background subtraction', fgmask)

		# Quit when escape key pressed
		if cv2.waitKey(5) == 27:
			break

		# Sleep to keep video speed consistent
		time.sleep(1.0 / FPS)

	# Clean up
	cap.release()
	cv2.destroyAllWindows()

	# remove all speeding_*.png images created in runtime
	for file in glob.glob('speeding_*.png'):
		os.remove(file)