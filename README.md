# Action-Recognition
Detect type and intensity of an action.

# requirements:
tensorflow 2.3
numpy

## Module Descriptions
# action-recognition
wraps a run loop which collects webcam frames and stores the extracted pose information in a person object.

# person
Holds all the information for a person which is extracted from a
webcam feed.

## Package Descriptions
# predictors
package containing:
- an Predictor ABC
- posepredictor module
- freqpredictor module
- actionpredictor module