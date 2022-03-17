import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random
import os

TEST_DIR = 'training/Test'
MODEL_FILE = 'model/default'
IMG_SIZE = (100, 100)

COLORS = np.array([
    (255,0,0),
    (255,0,255),
    (0,255,0),
    (0,127,255),
    (0,0,255),
    (127,255,0),
    (0,255,127),
    (255,127,0),
    (255,0,127),
    (255,255,0),
    (127,0,255),
    (0,255,255),
    (0,0,0),
    (0,0,0),
    (0,0,0),
    (0,0,0)
])

def main():
    vid = cv2.VideoCapture(0)

    # Load the Neural Network model
    loaded_model = load_model(MODEL_FILE)

    # # Load all frames of recording
    # test_files = [TEST_DIR + '/' + x for x in os.listdir(TEST_DIR)]

    # Create a gradient mask for the two colors
    gradient = np.tile(np.linspace(1.0, 0.5, IMG_SIZE[0]), (IMG_SIZE[1],1))

    # for test_file in test_files:
    while True:
        # Split the image left and right
        # img1, img2 = process_single_file(test_file)
        img1, img2 = process_single_video_frame(vid)

        # Format arrays correctly
        img1_ = np.array([img1]).reshape(-1, *IMG_SIZE, 1)
        img2_ = np.array([img2]).reshape(-1, *IMG_SIZE, 1)

        # Predict using the model
        pred1 = loaded_model.predict(img1_)
        pred2 = loaded_model.predict(img2_)

        # Use prediction as weights for color
        dot1 = np.dot(pred1, COLORS)
        dot2 = np.dot(pred2, COLORS)

        # Normalize colors
        col1 = (dot1 / dot1.max()) if dot1.max() != 0 else dot1
        col2 = (dot2 / dot2.max()) if dot2.max() != 0 else dot2

        # Create a frame to display
        frame = np.zeros((IMG_SIZE[1],IMG_SIZE[0]*2,3))

        # Color the left half
        frame[:,:IMG_SIZE[0],0] = img1 * (col1[0,0] * gradient + (col2[0,0] * (1 - gradient)))
        frame[:,:IMG_SIZE[0],1] = img1 * (col1[0,1] * gradient + (col2[0,1] * (1 - gradient)))
        frame[:,:IMG_SIZE[0],2] = img1 * (col1[0,2] * gradient + (col2[0,2] * (1 - gradient)))

        # Color the right half
        frame[:,IMG_SIZE[0]:,0][:,::-1] = img2 * (col2[0,0] * gradient + (col1[0,0] * (1 - gradient)))
        frame[:,IMG_SIZE[0]:,1][:,::-1] = img2 * (col2[0,1] * gradient + (col1[0,1] * (1 - gradient)))
        frame[:,IMG_SIZE[0]:,2][:,::-1] = img2 * (col2[0,2] * gradient + (col1[0,2] * (1 - gradient)))

        # Display the frame
        cv2.imshow('img', frame.astype(np.uint8))
        key = cv2.waitKey(16) & 0xff

        # Allow user to exit
        if key == ord('q'):
            break

def process_single_file(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return process_single_img(img)

def process_single_video_frame(vid):
    ret, img = vid.read()
    if ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return process_single_img(img)
    else:
        return process_single_img(np.zeros(IMG_SIZE[1], IMG_SIZE[0] * 2))

def process_single_img(img):
    img = cv2.resize(img, (2*IMG_SIZE[0], IMG_SIZE[1]))

    return (img[:,:IMG_SIZE[0]], img[:,IMG_SIZE[0]:][:,::-1])

if __name__ == "__main__":
    main()

# Sources:
# [1] https://note.nkmk.me/en/python-numpy-generate-gradation-image/