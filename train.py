import pandas as pd
import numpy as np
import wandb
import argparse
from fastai.vision.all import SimpleNamespace

# Preprocessing libraries
from sklearn.preprocessing import StandardScaler

# Splitting libraries
from sklearn.model_selection import train_test_split

# Training libraries
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Define default configuration for the experiment
default_config = SimpleNamespace(
    framework="fastai",
    seed=42,
    log_preds=False,
    data_path="data/Cancer_Data.csv",
    test_size=0.33,
    model_index=0
)

def parse_args() -> None:
    """
    Override default arguments with the ones provided by the user.
    """
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--seed', type=int, default=default_config.seed, help='random seed')
    argparser.add_argument('--log_preds', type=bool, default=default_config.log_preds, help='log model predictions')
    argparser.add_argument('--data_path', type=str, default=default_config.data_path, help='path to csv dataset')
    argparser.add_argument('--test_size', type=float, default=default_config.test_size, help='size of test split from data')
    argparser.add_argument('--model_index', type=int, default=default_config.model_index, help="list of index: 0(default)=LogisticRegression 1=RidgeClassifier 2=SGDClassifier 3=PassiveAggressiveClassifier 4=Perceptron")
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

def read_data(data_path: str) -> pd.DataFrame:
    """
    Read dataset from a CSV file.

    Args:
        data_path: A string representing the path to the CSV dataset file.

    Returns:
        A pandas DataFrame containing the dataset.
    """
    return pd.read_csv(data_path, index_col=0)

def preprocess(data: pd.DataFrame) -> tuple:
    """
    Preprocess the dataset.

    Args:
        data: A pandas DataFrame containing the dataset.

    Returns:
        A tuple containing the preprocessed features and target variable.
    """
    X = data.drop(columns=['Unnamed: 32', 'diagnosis'])
    y = data.diagnosis.map({"B":0, "M":1})
    X_t = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    return X_t, y

def split(X_t: pd.DataFrame, target: pd.Series, test_size: float, random_state: int) -> tuple:
    """
    Split the dataset into training and testing sets.

    Args:
        X_t: A pandas DataFrame containing the preprocessed features.
        target: A pandas Series containing the target variable.
        test_size: A float representing the proportion of the dataset to include in the test split.
        random_state: An integer to set the random seed for reproducibility.

    Returns:
        A tuple containing the training and testing sets.
    """
    return train_test_split(X_t, target, test_size=test_size, random_state=random_state)

def select_model(i: int = 0) -> object:
    """
    Select a model based on the index provided.

    Args:
        i: An integer representing the index of the desired model.

    Returns:
        A scikit-learn classifier object.
    """
    models = [LogisticRegression(), RidgeClassifier(), SGDClassifier(), PassiveAggressiveClassifier(), Perceptron()]
    try:
        return models[i]
    except:
        print("Error: model's index not in range. Set model_index=0.")
    return models[0]

def train(config: SimpleNamespace) -> None:
    """
    Train the selected model using the given configuration.
    Args:
        config: A SimpleNamespace object containing the experiment configuration.
    """
    data_path = config.data_path
    data = read_data(data_path=data_path)

    data_t, target = preprocess(data)
    X_train, X_test, y_train, y_test = split(data_t, target, config.test_size, config.seed)

    model = select_model(config.model_index)

    run = wandb.init(project="Assignment2", entity="saturdays", job_type="training", name=model.__str__()[:-2])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = np.round(mean_squared_error(y_test, y_pred), 6)
    r2 = np.round(r2_score(y_test, y_pred), 6)
    accuracy = accuracy_score(y_test, y_pred)
    wandb.log({'mse':mse, 'r2':r2, 'accuracy':accuracy})
    wandb.summary["mse"] = mse
    wandb.summary["r2"] = r2
    wandb.summary["accuracy"] = accuracy
    wandb.finish()

if __name__ == '__main__':
    parse_args()
    train(default_config)