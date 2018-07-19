FROM ubuntu:18.04

# Install Python3 with pip
RUN apt update && \
  apt -y upgrade && \
  apt -y install python3 git python-pip python-dev build-essential libsm6 libxext6 && \
  DEBIAN_FRONTEND=noninteractive apt -yq install python3-pip python3-tk
RUN pip3 install tensorflow opencv-python
RUN pip install opencv-python && pip install tensorflow

RUN mkdir /var/openpose
WORKDIR /var/openpose

RUN git clone https://www.github.com/ildoonet/tf-openpose
RUN cd tf-openpose && python3 setup.py install

CMD python3 tf-openpose/run_webcam.py
