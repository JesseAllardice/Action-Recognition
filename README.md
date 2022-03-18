# Action-Recognition
This project lets you detect the type and intensity of an action performed by a person in front of a webcam. The model uses **PoseNet** to prediction the key points of a human in every frame of a video. The values for these key points are then feed into a **convolution neural network** and a prediction of the action being performed is made.

Once the model is trained it can run live via a webcam.

The datasets for video-action pairs can be collected using the project [Action-Recording](https://github.com/JesseAllardice/Action_Recording).

# Requirements:
tensorflow 2.3
numpy

# Module Descriptions
### Action-recognition
wraps a run loop which collects webcam frames and stores the extracted pose information in a person object.

### Person
Holds all the information for a person which is extracted from a
webcam feed.

### Predictors
Package containing:
- an Predictor ABC
- posepredictor module
- freqpredictor module
- actionpredictor module