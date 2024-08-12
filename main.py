# Import necessary libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from evaluation import evaluate_classification
import pickle

""" 
    Returns dataframes for train, validation, test when proper path is provided
    Parameters - Path to any data (train, validation, test) 
    Example - df_train = load_data('wine_classification/train.csv'), df_validation= load_data('housing_price_prediction/df_validation.csv'), df_test = load_data('housing_price_prediction/df_test.csv')
"""
def load_data(path):
    df = pd.read_csv(path)

    return df

"""
    Returns a list of high skew features in given dataframe
    Parameters - Dataframe
    Example - skew_list_train = skewed_list(df_train)
"""
def skewed_list(df):
    skewed_list = [name for name in df if df[name].skew() > 1 or df[name].skew() < -1]

    return skewed_list

"""
    Removes duplicate rows in given dataframe
    Parameters - Dataframe
    Example - df_train = rem_dups(df_train)
"""
def rem_dups(df):
    df = df.drop_duplicates()

    return df

"""
    Applies Yeo-Johnson transformation on skewed features
    Parameters - Dataframe, list of skewed features
    Example - df_train = yeo_john(df_train, skew_list_train)
"""
def yeo_john(df, list):
    pt = PowerTransformer(method='yeo-johnson')
    for name in list:
        df[name] = pt.fit_transform(df[[name]])

    return df

"""
    Splits data into X and y
    Parameters - Dataframe
    Example - X_train, y_train = split_data(df_train)
"""
def split_data(df):
    X = df.drop("quality", axis = 1)
    y = df['quality']

    return X, y

"""
    Imputes outliers in given dataframe
    Parameters - Dataframe
    Example - df_train = impute_outliers(df_train)
"""
def impute_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median_value = df[column].median()
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), median_value, df[column])
    
    return df

"""
    Scales data using MinMaxScaler
    Parameters - Dataframe
    Example - X_train = mm_scaler(X_train), y_train = mm_scaler(y_train)
"""
def mm_scaler(df):
    scaler = MinMaxScaler()

    # Fit and transform the training data
    df_scaled = scaler.fit_transform(df)

    # Convert the scaled data back to DataFrame
    df = pd.DataFrame(df_scaled, columns=df.columns)

    return df

"""
    Drops residual sugar column
    Parameters - Dataframe
    Example - X_train = drop_rs(X_train)
"""
def drop_rs(df):
    df = df.drop(['residual sugar'], axis = 1)

    return df

"""
    Loads pickle file as model
    Parameters - Path
    Example - model = load_pickle('model.pkl')
"""
def load_pickle(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)

    print("Model loaded from pickle file.")

    return model

"""
    Generates predictions
    Parameters - Model
    Example - y_pred = prediction_generation(model, X_test), y_pred = prediction_generation(model, X_valid)
"""
def prediction_generation(model, X):
    y_pred = model.predict(X)

    return y_pred
