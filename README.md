# MoShNVR

## About
MoShNVR is a Neural Network-based mouth shape estimation software. It is mainly written in Python and was trained using Keras (in Tensorflow).

## Required Packages
- Numpy
- OpenCV
- Tensorflow==2.4.1
- MatPlotLib (optional)

## Potential Biases
A majority of the images used in the training dataset were of young adult white males that were clean-shaven. We tried to introduce more variation by including some images from [a Kaggle "yawn" dataset](https://www.kaggle.com/davidvazquezcic/yawn-dataset/metadata). However, these images make up less than 1% of the total training data.

## Running Instructions
To run the server, execute the following command from the root project folder
`python .`

A window should pop up with a preview of the user's webcam feed.

If you want to start the program with a video file instead of the webcam, use the optional `-i [VIDEO FILE]` parameter, where `[VIDEO FILE]` is the path to a video file.

Once the program starts up, click and drag on the preview image to specify where the mouth is located in the video stream. This will start streaming data over UDP on port 20001. The data should be in JSON format. If no region of the images has been specified, this data will be 'null.'

If you want to save this data to a file, use the optional `-o [FILE]` parameter, where `[FILE]` is the path to the desired output location. If the file does not exist yet, it will be created; otherwise, the program will append its output to the file.

If you want to preview the output data, run `python ./udp_graph.py` while the server is running.

## Known Issues
Sometimes the program will crash when a box is drawn. We believe that this has something to do with cuDNN or TensorFlow failing to initialize for some reason. However, we haven't been able to track down a fix for this.