import numpy as np
import cv2
import socket
import threading

# Set up constants
LOCAL_IP = "127.0.0.1"
LOCAL_PORT = 20001
BUFFER_SIZE = 1024
SCREEN_SIZE = (1280, 720)
WINDOW_NAME = 'MoShNVR'

# Set up global variables for callback functions
# (I really don't like this. I'm looking for a way around this)
global_drag = [(0,0), (0,0)]
global_box = [(0,0), (0,0)]
global_click = False

# MAIN
# Starting point for the program
def main():
    # Setup window with mouse support
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, 0)
    frame = np.zeros((*SCREEN_SIZE[::-1], 3), dtype=np.uint8)

    # Setup video stream
    vid = cv2.VideoCapture(0)

    # Setup UDP thread
    shared_data = [True, 0]
    lock = threading.Lock()
    udp_thread = threading.Thread(target=manage_udp, args=(shared_data, lock))
    udp_thread.start()

    # Main loop
    while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1: # [5]
        # Show Current Frame
        cv2.imshow(WINDOW_NAME, frame)

        # Exit if needed
        key_code = cv2.waitKey(16)
        if (key_code & 0xFF) == ord('q') or (key_code & 0xFF) == 27:
            cv2.destroyAllWindows()
            break

        # Update shared data
        with lock:
            shared_data[1] += 1
            print(shared_data[1])

    # Close UDP thread
    with lock:
        shared_data[0] = False
    udp_thread.join()

def get_frame(vid):
    pass

def manage_udp(shared_data, lock):
    # Set up UDP socket
    udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_socket.settimeout(1)
    udp_socket.bind((LOCAL_IP, LOCAL_PORT))

    # Send data as requested
    while(True):
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
            with lock:
                if not shared_data[0]:
                    break

# ON MOUSE
# Handle the mouse input for specifying mouth location in video stream
def on_mouse(event, x, y, flags, params): # [1]
    global global_drag
    global global_box
    global global_click

    if global_click == True:
        global_drag[1] = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        global_drag = [(x, y), (x,y)]
        global_box = [(x, y), (x,y)]
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



# Only run things if called directly
if __name__ == "__main__":
    main()

# Sources
# [1] https://stackoverflow.com/questions/22140880/drawing-rectangle-or-line-using-mouse-events-in-open-cv-using-python
# [2] https://note.nkmk.me/en/python-numpy-generate-gradation-image/
# [3] https://pythontic.com/modules/socket/udp-client-server-example
# [4] https://quick-adviser.com/can-multiple-udp-sockets-on-same-port/
# [5] https://medium.com/@mh_yip/opencv-detect-whether-a-window-is-closed-or-close-by-press-x-button-ee51616f7088