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

	def
