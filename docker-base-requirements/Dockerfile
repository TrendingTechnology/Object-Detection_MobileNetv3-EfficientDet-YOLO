FROM python:3.7

ENV HOME /app
COPY . ${HOME}
WORKDIR $HOME

# Install requirements
RUN apt-get update
RUN pip install -r requirements.txt
RUN python3 -m ipykernel install --user

# install OpenCV requirements
RUN apt-get install -yq cmake \
    gcc g++ \
    python3-dev python3-numpy \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
    libgtk-3-dev \

# OpenCV 4.3.0 installed manually inside the image
# https://docs.opencv.org/4.3.0/d2/de6/tutorial_py_setup_in_ubuntu.html

EXPOSE 8888

CMD ["bash"]