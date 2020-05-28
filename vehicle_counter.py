import math

import cv2
import numpy as np

# https://stackoverflow.com/questions/36254452/counting-cars-opencv-python-issue/36274515#36274515

class Vehicle (object):
	def __init__ (self, carid, position, start_frame):
		self.id = carid
		
		self.positions = [position]
		self.frames_since_seen = 0
		self.counted = False
		self.start_frame = start_frame

		self.speed = None

		# assign a random color for the car
		self.color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))

	@property
	def last_position (self):
		return self.positions[-1]
	
	def add_position (self, new_position):
		self.positions.append(new_position)
		self.frames_since_seen = 0

	def draw (self, output_image):

		for point in self.positions:
			cv2.circle(output_image, point, 2, self.color, -1)
			cv2.polylines(output_image, [np.int32(self.positions)], False, self.color, 1)

		if self.speed:
			cv2.putText(output_image, ("%1.2f" % self.speed), self.last_position, cv2.FONT_HERSHEY_PLAIN, 0.7, (127, 255, 255), 1)


class VehicleCounter (object):
	def __init__(self, shape, road, fps, samples=0):
		self.height, self.width = shape
		self.divider = road['divider']
		self.is_horizontal = road['divider_horizontal']
		self.pass_side = road['divider_pass_side']
		self.vector_angle_min = road['vector_angle_min']
		self.vector_angle_max = road['vector_angle_max']

		self.vehicles = []
		self.next_vehicle_id = 0
		self.vehicle_count = 0

		self.max_unseen_frames = 10

		self.sample_num = samples

		if samples == 0:
			print('DISTANCE MODE')
			self.distance = road['distance']
			self.fps = fps
		else:
			print('AVERAGE MODE')
			self.samples = []
			self.average_speed = -1
			self.average_threshold = 0.3

			self.average_distance = -1
			self.distances = []

	@staticmethod
	def get_vector (a, b):
		dx = float(b[0] - a[0])
		dy = float(b[1] - a[1])

		distance = math.sqrt(dx**2 + dy**2)

		if dy > 0:
			angle = math.degrees(math.atan(-dx/dy))
		elif dy == 0:
			if dx < 0:
				angle = 90.0
			elif dx > 0:
				angle = -90.0
			else:
				angle = 0.0
		else:
			if dx < 0:
				angle = 180 - math.degrees(math.atan(dx/dy))
			elif dx > 0:
				angle = -180 - math.degrees(math.atan(dx/dy))
			else:
				angle = 180.0

		return distance, angle

	@staticmethod
	def is_valid_vector (a, angle_min, angle_max):
		# TODO!
		distance, angle = a
		# TODO: This also needs to be customized!
		return (distance <= 60 and angle > angle_min and angle < angle_max)

	# This method should be customizeable/depend on external settings (i.e. horizontal vs vertical divider)
	# see explanation for pass_side in settings.json
	def is_past_divider (self, centroid):
		x, y = centroid

		if self.is_horizontal:
			if self.pass_side == -1:
				return y < self.divider
			else:
				return y > self.divider

		else:
			if self.pass_side == -1:
				return x < self.divider
			else:
				return x > self.divider


	def update_vehicle (self, vehicle, matches):
		# Find if any of the matches fits this vehicle
		for i, match in enumerate(matches):
			contour, centroid = match

			vector = self.get_vector(vehicle.last_position, centroid)
			if self.is_valid_vector(vector, self.vector_angle_min, self.vector_angle_max):
				print('Angle: %s' % vector[1])
				vehicle.add_position(centroid)

				return i

		# No matches fit
		# print('No matches found for vehicle %s' % vehicle.id)
		vehicle.frames_since_seen += 1

		return None


	# TODO: REMOVE small "ghost tracks" by comparing them to avg distance!!!
	def update_count (self, matches, frame_number, output_image=None):

		# Update existing vehicles
		for vehicle in self.vehicles:
			i = self.update_vehicle(vehicle, matches)
			if i is not None:
				del matches[i]

		# For remaining matches, add new vehicles
		# TODO: IMPORTANT: the bug of multiple little tracks on the same vehicle FIX THIS RONIT GODDAMN IT!!!
		for match in matches:
			contour, centroid = match
			
			# skip_this = False

			# for v in self.vehicles:
			# 	if v.last_position == centroid:
			# 		skip_this = True
			# 		break

			# if skip_this:
			# 	continue

			if not self.is_past_divider(centroid):
				new_vehicle = Vehicle(self.next_vehicle_id, centroid, frame_number)
				self.next_vehicle_id += 1
				self.vehicles.append(new_vehicle)

		# Count any uncounted vehicles that are past the divider
		for vehicle in self.vehicles:
			if not vehicle.counted and self.is_past_divider(vehicle.last_position):
				if self.sample_num == 0:
					# Distance mode

					# Running average of first 20-100 cars, use that as a benchmark (zeroing out the scale)
					time_alive = (frame_number - vehicle.start_frame)/self.fps
					# Convert to hours
					time_alive = time_alive / 60 / 60

					# MPH
					vehicle.speed = self.distance / time_alive

					# print(self.distance, time_alive)
				
				else:
					# Average mode
					distance = self.get_vector(vehicle.last_position, vehicle.positions[0])[0] # We don't need the angle

					speed = distance / (frame_number - vehicle.start_frame)
					print(f"SPEED: {speed}")		

					if len(self.samples) < self.sample_num:
						# Add to samples

						self.samples.append(speed)
						self.distances.append(distance)

						# Should we take the average now?
						if len(self.samples) == self.sample_num:
							self.average_speed = sum(self.samples)/len(self.samples)
							self.average_distance = sum(self.distances)/len(self.distances)

							print(f"AVERAGE SPEED: {self.average_speed}")

					else:
						# Throw it out if the distance is bizarrely long/short
						# if abs(distance-self.average_distance)/self.average_distance > 0.3:
						# 	print('FREAK DISTANCE DETECTED!')
						# 	continue

						speed_diff = (speed-self.average_speed)/self.average_speed
						vehicle.speed = speed_diff # Assuming average speed translates to 70 mph!

						if speed_diff >= self.average_threshold:
							print(f"{vehicle.id} is SPEEDING: {speed_diff}")

				self.vehicle_count += 1
				vehicle.counted = True


		# Draw the vehicles (optional)
		if output_image is not None:
			for vehicle in self.vehicles:
				vehicle.draw(output_image)

			cv2.putText(output_image, ("%02d" % self.vehicle_count), (0, 0), cv2.FONT_HERSHEY_PLAIN, 0.7, (127, 255, 255), 1)

		# Remove vehicles that have not been seen in a while
		removed = [v.id for v in self.vehicles
			if v.frames_since_seen >= self.max_unseen_frames]
		self.vehicles[:] = [v for v in self.vehicles
			if not v.frames_since_seen >= self.max_unseen_frames]
		# for carid in removed:
		# 	print('Removed vehicle', carid)

		# print('Count updated')