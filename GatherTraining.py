import numpy as np
import cv2

SCREEN_SIZE = (1280, 720)

global_drag = [(0,0), (0,0)]
global_box = [(0,0), (0,0)]
global_click = False

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


def main():
	cv2.namedWindow('Window')
	cv2.setMouseCallback('Window', on_mouse, 0)

	vid = cv2.VideoCapture(0)

	mouth_frame = np.zeros((100, 200))

	frame_number = 0

	while True:
		ret, frame = vid.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		frame = cv2.resize(frame, SCREEN_SIZE)

		mouth_width = global_box[1][0] - global_box[0][0]
		mouth_height = global_box[1][1] - global_box[0][1]
		
		if (mouth_width > 0) and (mouth_height > 0):
			mouth_frame = frame[global_box[0][1]:global_box[1][1], global_box[0][0]:global_box[1][0]]
			mouth_frame = cv2.resize(mouth_frame, (200, 100))

			# mouth_frame_1 = mouth_frame[:,:100]
			# mouth_frame_2 = mouth_frame[:,:-101:-1]

			# mouth_frame = (mouth_frame_1 // 2) + (mouth_frame_2 // 2)

		frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		frame = cv2.rectangle(frame, *global_drag, (0, 255, 0), 3)
		frame = cv2.rectangle(frame, *global_box, (255, 255, 0), 3)

		cv2.imshow('Window', frame)
		cv2.imshow('Mouth', mouth_frame)

		if cv2.waitKey(17) & 0xFF == ord('q'):
			break

		if frame_number > 120 and frame_number < 360:
			cv2.imwrite(f"training/001/{str(frame_number).zfill(5)}.png", mouth_frame)

		frame_number += 1

	vid.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()

# Sources
# [1] https://stackoverflow.com/questions/22140880/drawing-rectangle-or-line-using-mouse-events-in-open-cv-using-python