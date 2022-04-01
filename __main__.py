import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # [13]

from tensorflow.keras.models import load_model
import socket
import threading
import json
import argparse

# Set up constants
LOCAL_IP = "127.0.0.1"
LOCAL_PORT = 20001
BUFFER_SIZE = 1024

SCREEN_SIZE = (1280, 720)
MODEL_IMG_SIZE = (100, 100)
MOUTH_IMG_SIZE = (MODEL_IMG_SIZE[0] * 2, MODEL_IMG_SIZE[1])

WINDOW_NAME = 'MoShNVR'
MODEL_FILE_REL = 'model/default.h5'
LABELS_PATH_REL = 'mouth_shape_categories.txt'

DIRNAME = os.path.dirname(__file__) # [7]
MODEL_FILE = os.path.join(DIRNAME, MODEL_FILE_REL) # [7]
LABELS_PATH = os.path.join(DIRNAME, LABELS_PATH_REL) # [7]

BOUNDING_BOX_COLOR = (255, 255, 0)
DRAG_BOX_COLOR = (0, 255, 0)
CENTER_LINE_COLOR = (0, 255, 255)

# Set up global variables for callback functions
# (I really don't like this. I'm looking for a way around this)
global_drag = [(0,0), (0,0)]
global_box = [(0,0), (0,0)]
global_center_line = [(0,0), (0,0)]
global_click = False

# MAIN
# Starting point for the program
def main():
    # Handle command line arguments [10]
    parser = argparse.ArgumentParser(prog='MoShNVR',description='Tracks mouth shape, outputs to udp socket or optionally a specified file')
    parser.add_argument('-o', '--output', help = "Name and directoy of the file you want to save")
    parser.add_argument('-i', '--input', help = "Location of the video you want to process")
    args = parser.parse_args()

    file_path = args.output
    input_video = args.input

    # Load in the trained model
    model = load_model(MODEL_FILE)

    # Get Column Names
    labels = []
    with open(LABELS_PATH) as f:
        labels = [x.strip() for x in f]

    # Setup window with mouse support
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, 0)
    frame = np.zeros((*SCREEN_SIZE[::-1], 3), dtype=np.uint8)

    # Setup video stream
    if input_video is not None:
        vid = cv2.VideoCapture(input_video)
        video_started = False
    else:
        vid = cv2.VideoCapture(0)
        video_started = True

    if not vid.isOpened(): # [14]
        print("Failed to open video stream. Check that you have a webcam connected or a valid input file specified.")
        cv2.destroyAllWindows()
        return

    # Setup UDP thread
    shared_data = [True, 0]
    lock = threading.Lock()
    udp_thread = threading.Thread(target=manage_udp, args=(shared_data, lock))
    udp_thread.start()

    # Initialize variables
    prediction = None
    preview_enabled = True

    # Add '[' to output file to ensure valid JSON
    if file_path is not None:
        with open(file_path, 'a+') as f:
            f.write('[')

    # Main loop
    while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1: # [5]
        # Check if video should go to the next frame
        video_started = video_started or (global_box[0] != global_box[1])
        if not video_started:
            vid.set(1,0) # [11]

        # Get raw frame
        frame = get_raw_frame(vid, not video_started)
        if frame is None:
            break
        
        # Get display frame
        disp = get_display_frame(frame, preview_enabled)

        # Show Current Frame
        cv2.imshow(WINDOW_NAME, disp)
        # if mouth_frame is not None:
            # cv2.imshow("mouth", mouth_frame)

        # Exit if needed
        key_code = cv2.waitKey(16)
        if (key_code & 0xFF) == ord('q') or (key_code & 0xFF) == 27:
            break
        if (key_code & 0xFF) == ord(' '):
            preview_enabled = not preview_enabled

        # Get processing frame
        mouth_frame = get_mouth_img(frame)

        # Update Prediction
        if mouth_frame is not None:
            prediction = predict(model, labels, mouth_frame)

        # Update shared data
        with lock:
            shared_data[1] = json.dumps(prediction, indent=4) # [6]

        # Optionally write prediction to file
        if video_started and file_path is not None:
            with open(file_path, 'a+') as f:
                f.write(f'{shared_data[1]},')

    # Destroy all windows
    cv2.destroyAllWindows()
    
    # Add ']' to output file to ensure valid JSON
    if file_path is not None:
        with open(file_path, 'a+') as f:
            f.write('null]')

    # Close UDP thread
    with lock:
        shared_data[0] = False
    udp_thread.join()

def get_raw_frame(vid, video_started=True):
    # Read frame from video stream
    ret, frame = vid.read()

    if not ret:
        return None

    # Format and resize frame correctly
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, SCREEN_SIZE)

    return frame

def get_display_frame(frame, preview_enabled=True):
    # Optionally disable camera preview
    if not preview_enabled:
        return np.zeros((200,200,3))

    # Format frame correctly
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Draw Bounding Boxes
    frame = cv2.rectangle(frame, *global_drag, DRAG_BOX_COLOR, 3)
    frame = cv2.rectangle(frame, *global_box, BOUNDING_BOX_COLOR, 3)

    # Draw Center Line
    frame = cv2.rectangle(frame, *global_center_line, CENTER_LINE_COLOR, 3)

    return frame

def get_mouth_img(frame):
    # Find width and height
    mouth_width = global_box[1][0] - global_box[0][0]
    mouth_height = global_box[1][1] - global_box[0][1]

    # Initialize mouth_frame variable
    mouth_frame = None

    # If width and height are valid, get the frame
    if (mouth_width > 0) and (mouth_height > 0):
        mouth_frame = frame[global_box[0][1]:global_box[1][1], global_box[0][0]:global_box[1][0]]
        mouth_frame = cv2.resize(mouth_frame, MOUTH_IMG_SIZE)

    return mouth_frame

def process_single_img(img):
    return (img[:,:MODEL_IMG_SIZE[0]], img[:,MODEL_IMG_SIZE[0]:][:,::-1])

def predict(model, labels, img):
    # prediction = {'data':None}

    # Split the image left and right
    img_l, img_r = process_single_img(img)

    # Format arrays correctly
    arr_l = np.array([img_l]).reshape(-1, *MODEL_IMG_SIZE, 1)
    arr_r = np.array([img_r]).reshape(-1, *MODEL_IMG_SIZE, 1)

    # Make a prediction using the model
    pred_l = model.predict(arr_l)
    pred_r = model.predict(arr_r)

    # Combine and label predictions
    labels_l = [f'{label}_l' for label in labels]
    labels_r = [f'{label}_r' for label in labels]
    prediction = {k:v for k,v in zip(labels_l, pred_l[0].tolist())} # [8, 9]
    prediction.update({k:v for k,v in zip(labels_r, pred_r[0].tolist())}) # [8, 9]

    return prediction

def manage_udp(shared_data, lock):
    # Set up UDP socket
    udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_socket.settimeout(1)
    udp_socket.bind((LOCAL_IP, LOCAL_PORT))

    # Send data as requested
    while(shared_data[0]):
        try:
            # Get request from client
            _, addr = udp_socket.recvfrom(BUFFER_SIZE)

            # Setup response message
            with lock:
                message = str.encode(str(shared_data[1]))

            # Send reply to client
            udp_socket.sendto(message, addr)

        # Periodically check if UDP socket should close
        except socket.timeout:
            pass

# ON MOUSE
# Handle the mouse input for specifying mouth location in video stream
def on_mouse(event, x, y, flags, params): # [1]
    global global_drag
    global global_box
    global global_click
    global global_center_line

    if global_click == True:
        global_drag[1] = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        global_drag = [(x, y), (x,y)]
        global_box = [(x, y), (x,y)]
        global_center_line = [(x, y), (x,y)]
        global_click = True
    if event == cv2.EVENT_LBUTTONUP:
        global_click = False

        drag_width = abs(global_drag[0][0] - global_drag[1][0])
        drag_height = abs(global_drag[0][1] - global_drag[1][1])
        box_width = drag_width if drag_width > drag_height * 2 else drag_height * 2
        box_height = box_width // 2

        center_x = (global_drag[0][0] + global_drag[1][0]) // 2
        center_y = (global_drag[0][1] + global_drag[1][1]) // 2

        if box_width > SCREEN_SIZE[0]:
            box_width = SCREEN_SIZE[0]
            box_height = box_width // 2
        if box_height > SCREEN_SIZE[1]:
            box_height = SCREEN_SIZE[1]
            box_width = box_height * 2

        if center_x - (box_width // 2) < 0:
            center_x = box_width // 2
        if center_x + (box_width // 2) > SCREEN_SIZE[0]:
            center_x = SCREEN_SIZE[0] - box_width // 2

        if center_y - (box_height // 2) < 0:
            center_y = box_height // 2
        if center_y + (box_height // 2) > SCREEN_SIZE[1]:
            center_y = SCREEN_SIZE[1] - box_height // 2
        
        global_box = [(center_x - (box_width // 2), center_y - (box_height // 2)), (center_x + (box_width // 2), center_y + (box_height // 2))]
        global_center_line = [(center_x, global_box[0][1]), (center_x, global_box[1][1])]



# Only run things if called directly
if __name__ == "__main__":
    main()

# Sources
# [1] https://stackoverflow.com/questions/22140880/drawing-rectangle-or-line-using-mouse-events-in-open-cv-using-python
# [2] https://note.nkmk.me/en/python-numpy-generate-gradation-image/
# [3] https://pythontic.com/modules/socket/udp-client-server-example
# [4] https://quick-adviser.com/can-multiple-udp-sockets-on-same-port/
# [5] https://medium.com/@mh_yip/opencv-detect-whether-a-window-is-closed-or-close-by-press-x-button-ee51616f7088
# [6] https://www.geeksforgeeks.org/how-to-convert-python-dictionary-to-json/
# [7] https://stackoverflow.com/questions/918154/relative-paths-in-python
# [8] https://www.geeksforgeeks.org/python-merging-two-dictionaries/
# [9] https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tolist.html
# [10] https://docs.python.org/3/library/argparse.html
# [11] https://gist.github.com/vereperrot/dd2263e220e68555d687f2ed2075d590
# [12] https://stackoverflow.com/questions/20432912/writing-to-a-new-file-if-it-doesnt-exist-and-appending-to-a-file-if-it-does
# [13] https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
# [14] https://answers.opencv.org/question/200260/how-to-handle-opencv-warnings/