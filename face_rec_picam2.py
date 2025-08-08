#!/urs/bin/env python3
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
from threading import Thread
import queue

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

		# create directories if they don't exist
		os.makedirs(self.reference_images_dir, exist_ok=True)

		# Load existing database if available
		self.load_face_database()

	def load_image_and_encode(self, image_path):
		"""Load an image and create a face encoding"""
		try:
			# Load image
			image = face_recognition.face_locations(image_path)

			# find face locations
			face_locations = face_recognition.face_locations(image)

			if len(face_locations) == 0;
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
			if encoding is not None:
				self.known_face_encodings.append(encoding)
				self.known_face_names.append(person_name)

		# Save database
		self.save_face_database()
		print(f"Face database built with [len(self.known_face_encodings)} face encodings")


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
				self.known_dace_names = []
		else:
			print("No existing face database found")

	def setup_camera(self):
		"""Initialize camera using picamera2"""
		try:
			print("Initializing camera...")
			self.picam2 = Picamera2()

			# Configure camera
			camera_config = self.picam2.create_preview_configuration(
				main={"size": (640, 480), "format": "RGB888"}
			)
			self.picam2.configure(camera_config)

			# Start camera
			self.picam2.start()
			print("Camera initialized successfully")

			# let camera warm up for a couple of seconds
			time.sleep(2)
			return True

		except Exception as e:
			print(f"Error initializing camera: {e}")
			return False

	def capture_frame(self):
		"""Capture a frame from the camera"""
		try:
			if self.picam2 is None:
				return None:

			# Capture frame
			frame = self.picam2.capture_array()

			# Convert RGB to BGR for
			frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			return frame_bgr

		except Exception as e:
			print(f"Error capturing frame: {e}")
			returning None


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


