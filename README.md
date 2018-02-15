# opencv_smile
This repository contains a neural network that has been trained to recognize smiles. You can test it out yourself with either a webcam or a video file!

# Dependencies
Requires OpenCV 3.3.0

# Instruction on building files for Linux

First find a location on your computer for the repository. Then
```
1. git clone https://github.com/DevarakondaV/opencv_smile.git
2. cd opencv_smile
3. mkdir build
4. cd build
5. cmake ..
6. make
```
In order to run the executable you can either:

```
1. cd into opencv_smile/build/src
2. ./SmileNN [Options]
```
or: 

```
1. Create a folder with the name <your_folder_name>.
2. Create a two folders inside <your_folder_name> with the names "res" and "src"
3. Copy opencv_smile/build/res/haarcascade_frontalface_default.xml and opencv_smile/build/res/nn1.yml into <your_folder_name>/res
4. Copy opencv_smile/build/src/SmileNN into <your_folder_name>/src
5. ./SmileNN [Options]
```


# Using SmileNN
Using SmileNN is very simple because there are only two options. You can either use a webcam or a video file.

To use a video file, run the following in the terminal:
```
./SmileNN -f [path_to_video_file]
```
Note: The file has to be .avi format
To use a webcam:
```
1. cd \dev
2. find ttyVideo#
3. Remember the number "#" in ttyVideo#. It will mostlikely be 0.
4. cd to executable SmileNN
5. ./SmileNN -w #
```

# How does it work?
This implementation uses the viola jones method to find faces in each frame of an input stream. If a face is detected, it will then
select the part of the frame with the face, preprocess the image, and pass the data into a fully connected neural network.

# Limitations
There are some limitation to how well this implementation will work.
1. The accuracy of the neural network which predicts if a person is smiling or not is around 77.9%. This does not accounting for the pre-built cascade classifier which detects the faces. This accuracy is quite low
primary because of the model. Fully connected networks are not the best models for image classifcation. Furthermore, the size of the data set
used to train the model was quite small at approximately 2000 images.
2. It will not work correctly for faces that are upside down, in poor lighting conditions and faces that deviate to far from vertical. 
3. The resolution of the video will determine classification speed.

