import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def load_data():
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, "data", "insurance.csv")
    return pd.read_csv(path)


def preprocess(df):
    df = df.copy()

    # Encoding
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    df = pd.get_dummies(df, columns=['region'], drop_first=True)

    X = df.drop('charges', axis=1)
    y = df['charges']
    return X, y


def train_model():
    df = load_data()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test, df, X.columns