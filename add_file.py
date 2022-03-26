import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random
import os
import argparse
parser = argparse.ArgumentParser(prog='MoShNVR',description='Tracks mouth shape, outputs to udp socket or optionally a specified file')
parser.add_argument('-file_path', help = "Name and directoy of the file you want to save")
args = parser.parse_args()


file_path = args.file_path
if(file_path):
    with open(file_path, 'a+') as f:
        f.write('Hello World\n')
#https://stackoverflow.com/questions/20432912/writing-to-a-new-file-if-it-doesnt-exist-and-appending-to-a-file-if-it-does
