# Install OpenCV and face_recognition dependencies
echo "Installing dependencies..."

sudo apt install -y \
	build-essential cmake git unzip pkg-config \
	libjpeg-dev libpng-dev libtiff-dev \
	libavcodec-dev libavformat-dev libswscale-dev \
	libv4l-dev v4l-utils \
	libxvidcore-dev libx264-dev \
	libatlas-base-dev gfortran \
	python3-dev python3-pip python3-numpy \
	libopenblas-dev liblapack-dev \
	libboost-all-dev

echo "All dependencies installed successfully"
