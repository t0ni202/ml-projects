# Overview
This folder contains the machine learning project that focuses on analyzing historical exchange rates, ETFs tracking stock markets, and indices of US treasuries to predict the next hour's high of the Canadian dollar exchange rate using the scikit-learn library.

## Dataset

The dataset (appml-assignment1-dataset-v2.pkl) includes hourly data on:
Historical exchange rates against the US dollar
Trading prices of ETFs
An index of US treasuries, its volatility, and the S&P 500 index

## Tasks


Developing a predictive model

Preprocessing pipeline for data transformation

Model evaluation using the testing data set

Creating descriptive plots of the dataset

## Code Organization

code1.py and code2.py: Python scripts for creating the transformation pipelines and models.

model1.pkl and model2.pkl: Serialized versions of the trained machine learning models.

pipeline1.pkl and pipeline2.pkl: Serialized data transformation pipelines.

plotGeneration.py: Script for generating descriptive statistical plots saved as cad-change-stats.png.

## How to Use
Use model*.pkl and pipeline*.pkl for loading the trained models and preprocessing pipelines.

Run plotGeneration.py to generate and save the descriptive statistical plots.



