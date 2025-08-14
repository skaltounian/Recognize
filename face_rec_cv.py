#!/usr/bin/env python3
"""
Refactoring the face recognition system
"""

import cv2
import face_recognition
import numpy as nm
import os
import pickle
import time
from datetime import datetime
from picamera2 import Picamera2
from libcamera import Transform  # -- See if you can do without this
from threading import Thread
import queue
import warnings

class CameraFunctions:
	def __init__(self):
		"""Camera initialization"""
		self.picam2 = None
		self.frame_queue = queue.Queue(maxsize=2)
		self.camera_running = False
		self.image_rotation = 180  # -- Avoid using this parameter

	def setup_camera(self):
		"""Initialize camera"""
		print("Initialize camera.")

		try:
			print("Initializing camera...")
			self.picam2 = Picamera2()

			# Configure camera rotation

			# Configure camera
			camera_config = self.picam2.create_preview_configuration(
				main={"size": (640, 480), "format": "RGB888"},
				transform = Transform(rotation=self.image_rotation)
			)

			self.picam2.configure(camera_config)

			# Start camera and let it settle for 2 seconds
			self.picam2.start()
			print("Camera initialized successfully")
			time.sleep(2)
			return True

		except Exception as e:
			print(f"Error initializing camera: {e}")
			return False

	def capture_frame(self):
		"""Capture a frame from the camera"""
		print("Capture a frame from the camera.")

	def camera_thread(self):
		"""Thread function for continuous frame capture"""
		print("Thread function.")


class CVFunctions:
	def __init__(self):
		"""CV initialization"""
		self.known_face_encodings = []
		self.known_face_names = []
		self.face_database_path = "face_database.pkl"
		self.reference_images_dir = "reference_images"

		# Create directories if they don't exist
		os.makedirs(self.reference_images_dir, exist_ok=True)

		# Load existing database if available
		self.load_face_database()

		# instantiate the camera functions
		self.cam = CameraFunctions()  # Probably not necessary here.


	def load_image_and_encode(self, image_path):
		"""Load image and create face encoding"""
		print(f"Load image from {image_path} and encode face.")

		try:
			# Load image
			image = face_recognition.load_image_file(image_path)

			# Find face location
			face_locations = face_recognition.face_locations(image)
			if len(face_locations) == 0:
				print(f"No face found in {image_path}")
				return None
			elif len(face_locations) == 1:
				print(f"Found a face in {image_path}, using first one")
			elif len(face_locations) > 1:
				print(f"Found multiple faces in {image_path}, using first one")
			else:
				print("How did we end up here?")

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
		print("Build face database from reference images.")

		# Clear existing data
		self.known_face_encodings = []
		self.known_face_names = []

	# Process all images in the reference directory
		for filename in os.listdir(self.reference_images_dir):
			if filename.lower().endswith(('.png', '.jpeg', '.jpg')):
				# full image_directory-name + file-name
				image_path = os.path.join(self.reference_images_dir, filename)

				# Determine the person based on the file name
				filename_lower = filename.lower()
				if 'mandela' in filename_lower:
					person_name = "Nelson Mandela"
				elif 'carter' in filename_lower:
					person_name = "Jimmy Carter"
				else:
					print(f"Unknown person in filename: {filename}")
					continue

				# Encode face of known person
				encoding = self.load_image_and_encode(image_path)
				print(f"Completed encoding and image of {person_name}")
				if encoding is not None:
					self.known_face_encodings.append(encoding)
					self.known_face_names.append(person_name)

				# Save database
				self.save_face_database()
				print(f"Face database built with {len(self.known_face_encodings)} face encodings")

	def save_face_database(self):
		"""Save face database to file"""
		print("Save face database.")

		database = {
			'encodings': self.known_face_encodings,
			'names': self.known_face_names
		}

		with open(self.face_database_path, 'wb') as f:
			pickle.dump(database, f)
		print(f"...Face database saved to {self.face_database_path}")

	def load_face_database(self):
		"""Load face database from file"""
		print("Load face database from file.")

	def recognize_face(self, frame, confidence_threshold=0.6):
		"""Recognize faces in a frame"""
		print("Recognize faces in a frame.")

	def draw_results_on_frame(self, frame, results):
		"""Draw recognition results on frame"""
		print("Draw recognition results on frame.")

	def run_face_recognition(self):
		"""Run real-time face recognition"""
		print("Run real-time face recognition.")

		print("Controls:")
		print("   SPACE: Analyze current frame")
		print("   'q': Quit program")
		print("   'r': Rebuild face database")

		# Initialize camera
		if not self.cam.setup_camera():
			print("Failed to initialize the camera")
			return
		else:
			print("Initialized the camera")

def main():
	"""Main function"""
	print("=" * 50)
	print("Starting Face Recognition with Picamera2 and OpenCV")
	print("=" * 50)

	# Instantiate classes:
	cvf = CVFunctions()

	# Check if we have reference images
	if len(os.listdir(cvf.reference_images_dir)) == 0:
		print("Could not find reference images.")
		print("Please add reference images to the designated folder.")
		return
	else:
		print(f"Found reference images in {cvf.reference_images_dir}")

	if len(cvf.known_face_encodings) == 0:
		print("Face database not found. Building anew from reference images...")
		cvf.build_face_database()

		if len(cvf.known_face_encodings) == 0:
			print("No faces could be encoded. Please check reference images.")
			return
		else:
			print("Faces have been encoded successfully.")

	# Start camera for face recognition
	cvf.run_face_recognition()

if __name__ == "__main__":
	main()
