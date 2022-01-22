import numpy
import cv2

def main():
	vid = cv2.VideoCapture(0)

	while True:
		ret, frame = vid.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		frame = cv2.resize(frame, (1280, 720))

		cv2.imshow('frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


	vid.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()