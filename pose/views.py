import argparse
import logging
import time

import cv2
import numpy as np

from django.shortcuts import render
from django.http import HttpResponse
from pose.tf_pose_estimation import run_webcam

def index(request):
    run_webcam.webcam()
    return HttpResponse("Hello, world. You're at the index.")
