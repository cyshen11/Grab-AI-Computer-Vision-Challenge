# AI for Recognizing Vehicles' Make and Model

This is a data model to recognize vehicles' make and model from images.\
The CNN architecture used is ResNet with 34 layers.\
Currently the trained model has 46% validation accuracy.\
This model is created to participate in Grab's AI Challenge for SEA.

[Link to dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

## Getting Started

- Download the entire folder as zip into your local machine.
- Download [model_weights](https://drive.google.com/open?id=1bAsH38O1aY5oWb_GBKGUtHa4Vztj4Jkt) and place in the previously downloaded folder location

### Prerequisites

This model is created using Python 3.\
Python libraries needed are:
- numpy
- pandas
- os
- scipy
- keras
- cv2

## Use the trained model to predict images

1. Run Predict.py.
2. Enter test dataset folder path. (Example: C:/Users/HP/Documents/Machine Learning/Grab_AI_Challenge/test/)
3. Prediction result is printed.

## Built With

* Python 3

## Authors

* **Cheng Yun Sheng** - *Initial work* - [cyshen11](https://github.com/cyshen11)

## Acknowledgments

* Faisal Shahbaz for the ResNet python code.

