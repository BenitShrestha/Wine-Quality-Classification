import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from main import load_data, rem_dups, skewed_list, yeo_john, split_data, impute_outliers, mm_scaler, drop_rs, prediction_generation, load_pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from evaluation import evaluate_classification
import pickle

# Example usage

def main():
    # Load train, validation data as dataframes
    df_train = load_data("wine_classification/train.csv")
    df_test = load_data('wine_classification/wine-test-set.csv')

    # Remove duplicates
    df_train = rem_dups(df_train)
    df_test = rem_dups(df_test)

    # Retrieve list of skewed features
    skew_list_train = skewed_list(df_train)
    skew_list_test = skewed_list(df_test)

    # Yeo Johnson tranformation 
    df_train = yeo_john(df_train, skew_list_train)
    df_test = yeo_john(df_test, skew_list_test)

    # Split into input features(X) and target features(y)
    X_train, y_train = split_data(df_train)
    X_test, y_test = split_data(df_test)

    # Imputation
    for col in X_train.columns:
        X_train = impute_outliers(X_train, col)
    # X_train = impute_outliers(X_train, 'chlorides') # Due to high outliers

    for col in X_test.columns:
        X_test = impute_outliers(X_test, col)

    # Min Max Scaling
    X_train = mm_scaler(X_train)
    X_test = mm_scaler(X_test)

    # Remove residual sugar feature
    X_train = drop_rs(X_train)
    X_test = drop_rs(X_test)

    ''' MULTI CLASS CLASSIFICATION '''
    # Load the trained model
    model = load_pickle('Saved_Models/rfc_model_normal.pkl')

    ''' Generate predictions based on the trained model '''
    y_pred = prediction_generation(model, X_test)

    ''' Evaluate the predictions '''
    metrics = evaluate_classification(y_test, y_pred)
    for metric, value in metrics.items():
        if metric != 'Confusion Matrix':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}:\n{value}")
    
    ''' ---------------------------------------------- '''

    ''' BINARY CLASSIFICATION '''
    # Labeling the quality based on value counts, this is done for binary classification
    y_train.replace([4,5], 'bad', inplace = True)
    y_train.replace([6,7,8], 'good', inplace = True)

    y_test.replace([4,5], 'bad', inplace = True)
    y_test.replace([6,7,8], 'good', inplace = True)

    # Load the trained model
    model = load_pickle('Saved_Models/rfc_model_binary.pkl')

    ''' Generate predictions based on the trained model '''
    y_pred =  prediction_generation(model, X_test)

    y_pred = np.where(np.isin(y_pred, [4, 5]), 'bad', y_pred)    
    y_pred = np.where(np.isin(y_pred, [6, 7, 8]), 'good', y_pred)    
    y_pred = pd.DataFrame(y_pred)
    ''' Evaluate the predictions '''
    metrics = evaluate_classification(y_test, y_pred)
    for metric, value in metrics.items():
        if metric != 'Confusion Matrix':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}:\n{value}")

if __name__ == '__main__':
    main()