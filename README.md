# EN.585.771.81.SP26 Capstone Project - Steven Qian

## Breast Cancer Neural Network Model + Predictor

This is an interactive Streamlit application that trains a neural network using a dataset from the University Wisconsin, where 3 measurements of 10 features were taken of a fine needle aspirate (FNA) image of breast masses. This dataset then classifies these breast tumors as **malignant** or **benign**. This Streamlit application allows the user to control how to train a model off this data, and then offers a predictor, allowing the user to import or manually input measurement data of a new mass and predict whether that mass is **malignant** or **benign** with an associated **confidence score**.

---

## Dataset

The app uses the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) from the UCI Machine Learning Repository, loaded directly from the web at runtime. It contains **569 patient samples**, each described by **30 numeric features**.

The 30 features represent 10 nuclear characteristics, which are detailed in the dataset link above.

This dataset was chosen because it is small enough to allow many iterations of testing (the purpose of this application) and it is a complete dataset, eliminating further complexities that can come from incomplete data. 

---

## Model Architecture

The model is a neural network built in PyTorch.

It uses a linear layer with one hidden layer, batch normalization, ReLU activation, and uses BCEWithLogitsLoss to calculate loss values. 

**Training details:**
- Loss function: `BCEWithLogitsLoss`
- Optimizer: Adam
- Input scaling: `StandardScaler`

---

## Features

### Training
- Adjust all hyperparameters from the sidebar before training
- Live loss curve updates during training (training loss in blue, validation loss in red)
- Final training and validation loss displayed as metrics after each run
- Training run history table stores all parameter combinations and results for comparison

**Recommended starting point:** 100 epochs, learning rate 0.01, 16 neurons, dropout 0.2, 80/20 split.

**Target loss values:** Training and validation loss below 0.10, with a gap between them under 0.05.

### Predictor
Two input methods are available after training:

**CSV Upload**
- Upload a CSV file with a complete set of 30 columns for the 30 features
- A downloadable template is provided
- A sample file (`sample_patients.csv`) with 20 real patient records (10 malignant, 10 benign) is included in this directory for testing.

**Manual Input**
- Enter all 30 feature values directly in the app
- Fields are organized by measurement type (Mean, Standard Error, Worst)
- Defaults to dataset averages; bounded by dataset min/max

---

## Running the app

Install dependencies:

```
streamlit
pandas
torch
scikit-learn
```

```bash
pip install streamlit pandas torch scikit-learn
```

To run the app

```bash
streamlit run capstone_app.py
```

Connect to the local/network host endpoint output by Streamlit
