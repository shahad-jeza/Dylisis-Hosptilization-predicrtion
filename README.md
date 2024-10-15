# Hospitalization Prediction with GAN-Based Data Augmentation

## Overview
This project focuses on generating a synthetic dataset for hospitalization prediction and using a Generative Adversarial Network (GAN) to address data imbalance by augmenting the minority class (hospitalized patients). The dataset is generated with features commonly associated with hospitalization, such as age, gender, diabetes, hypertension, BMI, albumin, and hemoglobin levels.

## Features
- **Synthetic Data Generation**: Simulates patient data for both hospitalized and non-hospitalized individuals with a 2:1 ratio.
- **Data Preprocessing**: Standardizes features and separates the minority class (hospitalized patients) for augmentation.
- **GAN Implementation**: A GAN is trained to generate additional synthetic samples for the minority class to balance the dataset and improve predictive model performance.

## Dataset
The dataset includes:
- **Age**: Continuous variable representing patient age.
- **Gender**: Binary variable (0 = female, 1 = male).
- **Diabetes**: Binary variable (0 = no, 1 = yes).
- **Hypertension**: Binary variable (0 = no, 1 = yes).
- **BMI**: Body Mass Index.
- **Albumin**: Continuous variable for albumin levels.
- **Hemoglobin**: Continuous variable for hemoglobin levels.
- **Hospitalization**: Target variable (0 = no, 1 = yes).

## How to Use

1. **Install dependencies**:
   ```bash
   pip install numpy pandas scikit-learn tensorflow

<img width="1439" alt="Screenshot 1445-11-19 at 7 25 45 PM" src="https://github.com/user-attachments/assets/e6f6f760-d8be-48be-96fd-922e40cb33de">
