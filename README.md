# Reliability Test and Improvement of a Sensor System for Object Detection

## Overview
This project is dedicated to advancing the dependability of an object detection sensor system. It leverages diverse machine learning algorithms and data processing techniques to refine the sensor's detection accuracy.

## Installation Instructions

 **Cloning the repository:**
To begin, clone the project repository from its source:
```bash
   git clone <repository-url>
```
Ensure that your system has Python 3.9 or later installed. If not, please download the latest version from the Python official website.

Installing Dependencies
```bash
   pip install -r requirements.txt
```

## Script Overview

- **XG_Boost.py**: This script deploys an XGBoost algorithm for analyzing sensor data. It encompasses data pre-processing, model training, and performance evaluation.

- **ADC_FFT_Plot.py**: Dedicated to ADC data processing, this script utilizes FFT for frequency domain analysis and includes data visualization features.

- **CNN_Model.py**: Constructs and trains a Convolutional Neural Network (CNN) tailored for object detection with sensor data. It covers data preprocessing and model assessment.

- **Evaluation_Metrics.py**: Supplies various metrics for evaluating the accuracy and performance of machine learning models.

- **Merge_CSV.py**: This utility script amalgamates several CSV files into a single file, streamlining data management.

- **Random_Forest.py**: Utilizes a Random Forest algorithm for sensor data examination, including aspects of model training and evaluation.

- **test_model.py**: Assesses the efficacy of a pre-trained model on novel datasets, incorporating data preprocessing and performance evaluation.

- **TXT_to_CSV_converter.py**: Converts textual data files into CSV format, simplifying the data preprocessing stage.

## Usage Guidelines
Scripts are designed to be executed independently, catering to specific analysis or training requirements. For instance, to initiate training using the CNN model script, enter:
```bash
   python CNN_Model.py
```
Ensure that all requisite data files are appropriately placed in the expected directories as per each script's requirements.
