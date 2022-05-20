![badge](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

# Covid19-Prediction-RNN-LSTM

# 1. Summary 

The purpose of this project is to create a deep learning model using LSTM neural  network to predict new cases (cases_new) in Malaysia using the past 30 days of number of cases.

# 2. Dataset

This projects is trained with [Heart Attack Analysis & Prediction Dataset](https://github.com/MoH-Malaysia/covid19-public). However, some data has been modified to practice with data cleaning.

# 3. Requirements

This project is created using Spyder as the main IDE. The main frameworks used in this project are Pandas, Numpy, Sklearn, TensorFlow and Tensorboard.

# 4. Methodology
This project contains two .py files. The training and modules files are training.py and modules.py respectively. The flow of the projects are as follows:

## 1. Importing the libraries and dataset

The data are loaded from the dataset and usefull libraries are imported.

## 2. Exploratory data analysis

The datasets is cleaned with necessary step. The non numeric characters are removed and split. The empty cell is then imputed with Simple Imputer.

## 3. LSTM model

A Long Short-Term Memory (LSTM) model is created with layers  ≤ 64 and epochs=20. The raining loss is displayed using TensorBoard

![](https://github.com/ainnmzln/Covid19-Prediction-RNN-/blob/main/images/tensorboard.png)

## 4. Model Prediction and Accuracy

The mean absolute percentage error (MAPE) is then calculated against the testing dataset. The graph between actual and predicted covid-19 cases is the plotted.

![](https://github.com/ainnmzln/Covid19-Prediction-RNN-/blob/main/images/actual%20vs%20predicted.png)

# Conclusion

This project manages to achieve MAPE= 020% which is lesser than 1% by using the deep learning model developed
