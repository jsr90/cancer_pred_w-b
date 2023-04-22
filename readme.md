# Comparing Machine Learning Models for Cancer Prediction using Weights & Biases
This repository contains code for training predictive models on the Cancer_Dataset using Weights & Biases (wandb) for tracking experiments.

## Table of Contents

1. [About Saturdays.AI](#about-saturdaysai)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Training](#training)
5. [Conclusion](#conclusion)
6. [Acknowledgements](#acknowledgements)

## About Saturdays.AI

This project is a part of my internship at Saturdays.AI, an organization dedicated to democratizing Artificial Intelligence education and fostering a global community of AI practitioners. By participating in the internship, I had the opportunity to apply the concepts and techniques learned during the program to real-world problems, such as the cancer prediction task in this repository.


## Dataset
The dataset can be found in the data/Cancer_Dataset.csv file. It contains features related to cell nuclei extracted from breast cancer samples. The target variable is the diagnosis (benign or malignant).

## Exploratory Data Analysis (EDA)
A superficial exploratory data analysis has been performed in the EDA.ipynb notebook. The analysis focused on understanding the types of features (all continuous numerical variables) and the target variable (categorical, with 'B' for benign and 'M' for malignant).

## Training
Model training is carried out in the train.py script. To run the training, open a terminal and execute the following command:

```
python train.py
```

You can customize the training by providing arguments to the parse_args() function in the script. The available arguments are:

--seed: The random seed used for reproducibility (default: 42)
--log_preds: Whether or not to log model predictions (default: False)
--data_path: Path to the CSV dataset file (default: "data/Cancer_Data.csv")
--test_size: Proportion of the dataset to be included in the test split (default: 0.33)
--model_index: Index of the model to use for training (default: 0). The available models are:
0: LogisticRegression
1: RidgeClassifier
2: SGDClassifier
3: PassiveAggressiveClassifier
4: Perceptron
For example, to train a RidgeClassifier model, you can run:

```
python train.py --model_index 1
```

This will override the default model (LogisticRegression) and train a RidgeClassifier model instead.

## Conclusion
By leveraging Weights & Biases for tracking experiments, we were able to train and compare multiple models on the Cancer Dataset. This approach enables efficient model selection and aids in improving the overall performance of the predictive models. As you continue working on this project, consider exploring additional models, feature engineering techniques, or hyperparameter tuning to further improve the results.

## Acknowledgements
I would like to express my gratitude to Saturdays.AI, where I had the opportunity to participate in a practical learning experience. Their support and guidance have been instrumental in my growth and the development of this project.
