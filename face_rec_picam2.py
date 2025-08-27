#!/usr/bin/env python3
"""
Face recognition system - phase 1 using picamera2
"""

import cv2
import face_recognition
import numpy as np
import os
import pickle
import time
from datetime import datetime
from picamera2 import Picamera2
from libcamera import Transform
from threading import Thread
import queue
import warnings

class PiCamera2FaceRecognition:
	def __init__(self):
		self.known_face_encodings = []
		self.known_face_names = []
		self.face_database_path = "face_database.pkl"
		self.reference_images_dir = "reference_images"

		# camera setup
		self.picam2 = None
		self.frame_queue = queue.Queue(maxsize=2)
		self.camera_running = False
		self.image_rotation = 180  # Rotate image to have the right side up

		# create directories if they don't exist
		os.makedirs(self.reference_images_dir, exist_ok=True)

		# Load existing database if available
		self.load_face_database()

	def load_image_and_encode(self, image_path):
		"""Load an image and create a face encoding"""
		print("... load image and encode")

		try:
			# Load image
			image = face_recognition.load_image_file(image_path)

			# find face locations
			face_locations = face_recognition.face_locations(image)

			if len(face_locations) == 0:
				print(f"No face found in {image_path}")
				return None

			if len(face_locations) > 1:
				print(f"Multiple faces found in {image_path}, using the first one")

			# Get face encoding
			face_encodings = face_recognition.face_encodings(image, face_locations)

			if len(face_encodings) > 0:
				print(f"Successfully encoded face from {image_path}")
				return face_encodings[0]
			else:
				print(f"Could not encode face from {image_path}")
				return None
		except Exception as e:
			print(f"Error processing {image_path}: {e}")
			return None

	def build_face_database(self):
		"""Build face database from reference images"""
		print("Building face database...")

		# Clear existing data
		self.known_face_encodings = []
		self.known_face_names = []

		# Process all images in reference directory
		for filename in os.listdir(self.reference_images_dir):
			if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
				image_path = os.path.join(self.reference_images_dir, filename)

				# Determine person name from filename
				filename_lower = filename.lower()
				if 'mandela' in filename_lower:
					person_name = "Nelson Mandela"
				elif 'carter' in filename_lower:
					person_name = "Jimmy Carter"
				else:
					print(f"Unknown person in filename: {filename}")
					continue

				# Get face encoding
				encoding = self.load_image_and_encode(image_path)
				print("Completed encoding")
				if encoding is not None:
					self.known_face_encodings.append(encoding)
					self.known_face_names.append(person_name)

		# Save database
		self.save_face_database()
		print(f"Face database built with {len(self.known_face_encodings)} face encodings")


	def save_face_database(self):
		"""Save face database to file"""
		database = {
			'encodings': self.known_face_encodings,
			'names': self.known_face_names
		}
		with open(self.face_database_path, 'wb') as f:
			pickle.dump(database, f)
		print(f" Face database saved to {self.face_database_path}")

	def load_face_database(self):
		"""Load face database from file"""
		if os.path.exists(self.face_database_path):
			try:
				with open(self.face_database_path, 'rb') as f:
					database = pickle.load(f)
				self.known_face_encodings = database['encodings']
				self.known_face_names = database['names']
				print(f"Loaded face database with {len(self.known_face_encodings)} encodings")
			except Exception as e:
				print(f"Error loading face database: {e}")
				self.known_face_encodings = []
				self.known_face_names = []
		else:
			print("No existing face database found")

	def setup_camera(self):
		"""Initialize camera using picamera2"""

		try:
			print("Initializing camera...")
			self.picam2 = Picamera2()

			# Configure camera
			camera_config = self.picam2.create_preview_configuration(
				main={"size": (640, 480), "format": "RGB888"},
				transform = Transform(rotation=self.image_rotation),
			)

			self.picam2.configure(camera_config)

			# Start camera and let it warm up settle for 2 seconds
			self.picam2.start()
			print("Camera initialized successfully")
			time.sleep(2)

			return True

		except Exception as e:
			print(f"Error initializing camera: {e}")
			return False

	def capture_frame(self):
		"""Capture a frame from the camera"""

		try:
			if self.picam2 is None:
				return None

			# Capture and return frame
			frame = self.picam2.capture_array()

			return frame

		except Exception as e:
			print(f"Error capturing frame: {e}")
			return None


	def camera_thread(self):
		"""Thread function for continuous frame capture"""
		while self.camera_running:
			frame = self.capture_frame()
			if frame is not None:
				# Keep only the latest frame
				if not self.frame_queue.empty():
					try:
						self.frame_queue.get_nowait()
					except queue.Empty:
						pass
				try:
					self.frame_queue.put_nowait(frame)
				except queue.Full:
					pass
			time.sleep(0.033)  # ~30 FPS

	def recognize_face(self, rgb_frame, confidence_threshold=0.6):
		"""Recognize faces in a frame"""

		# Find face locations and encodings
		face_locations = face_recognition.face_locations(rgb_frame)
		face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

		if len(face_locations) == 0:
			print("No faces found")
			return []

		results = []

		for face_encoding, face_location in zip(face_encodings, face_locations):
			# Compare with known faces
			if len(self.known_face_encodings) > 0:
				face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
				best_match_index = np.argmin(face_distances)

				# Check if the best match is within threshold
				if face_distances[best_match_index] <= confidence_threshold:
					name = self.known_face_names[best_match_index]
					confidence = 1 - face_distances[best_match_index]  # Convert distance to confidence
				else:
					name = "Unknown Person"
					confidence = 0.0
			else:
				name = "Unknown Person"
				confidence = 0.0

			results.append({
				'name': name,
				'confidence': confidence,
				'location': face_location
			})
		return results

	def draw_results_on_frame(self, frame, results):
		"""Draw recognition results on frame"""
		for result in results:
			top, right, bottom, left = result['location']
			name = result['name']
			confidence = result['confidence']

			# Draw rectangle around face
			color = (0, 255, 0) if name != "Unknown Person" else(0, 0, 255)
			cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

			# Draw label
			label = f"{name}  ({confidence:0.2%})" if confidence > 0 else name
			cv2.rectangle(frame, (left, top - 35), (right, bottom), color, 2)
			cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
		return frame


	def run_face_recognition(self):
		"""Run real-time face recognition with picamera2"""
		print("Starting face recognition system...")
		print("Controls:")
		print("   SPACE: Analyze current frame")
		print("   'q': Quit")
		print("   'r': Rebuild face database")

		# Initialize camera
		if not self.setup_camera():
			print("Failed to initialize camera")
			return

		# Start camera thread
		self.camera_running = True
		camera_thread = Thread(target=self.camera_thread)
		camera_thread.daemon = True
		camera_thread.start()

		analyzing = False
		last_analysis_time = 0

		try:
			while True:
				# Get latest frame
				try:
					frame_rgb = self.frame_queue.get_nowait()
				except queue.Empty:
					# use a black frame if no frame available
					frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

				# Show current frame
				display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

				# Add current frame
				status_text = "Press SPACE to analyze" if not analyzing else "Analyzing..."
				cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

				cv2.imshow('Face Recognition System', display_frame)

				# Key presses
				key = cv2.waitKey(1) & 0xFF

				if key == ord('q'):
					break
				elif key == ord(' '):  # Space key
					if not analyzing and time.time() - last_analysis_time > 1:  # prevent spam
						analyzing = True
						print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Analyzing frame...")

						# save the frame being analyzed for debugging
						start_time = time.time()
						results = self.recognize_face(frame_rgb)
						processing_time = time.time() - start_time

						print(f"Processing time: {processing_time:.2f} seconds")

						if results:
							for i, result in enumerate(results):
								print(f"Face {i+1}: {result['name']} (Confidence: {result['confidence']:.2%})")
						else:
							print("No faces detected")

						# show results on frame for 3 seconds
						result_frame = self.draw_results_on_frame(frame_rgb.copy(), results)

						end_time = time.time() + 3
						while time.time() < end_time:
							cv2.imshow('Face Recognition System', result_frame)
							if cv2.waitKey(30) & 0xFF == ord('q'):
								break

						analyzing = False
						last_analysis_time = time.time()
				elif key == ord('r'):
					print("Rebuilding face database...")
					self.build_face_database()

		except KeyboardInterrupt:
			print("Interrupted by user")

		finally:
			# Cleanup
			self.camera_running = False
			if self.picam2:
				self.picam2.stop()
			cv2.destroyAllWindows()
			print("Camera stopped and windows closed")

def main():
	"""Main function"""
	print("Face Recognition System with Picamera2 and OpenCV")
	print("=" * 55)

	# Initialize system
	system = PiCamera2FaceRecognition()

	# Check if we have reference images
	if len(os.listdir(system.reference_images_dir)) == 0:
		print("Could not find reference images!")
		print("Please add reference images to the designated folder:")
		print("- mandela_1, etc.")
		print("- carter_1, ect.")
		return
	else:
		print("Found reference images in {system.reference_images_dir}!")

	# Build face database if needed
	if len(system.known_face_encodings) == 0:
		print("Face database not found. Building from reference images...")
		system.build_face_database()

	if len(system.known_face_encodings) == 0:
		print("No faces could be encoded. Please check reference images.")
		return

	# Start camera recognition
	system.run_face_recognition()

if __name__ == "__main__":
	main()
