#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import shap
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import argparse


def run_shap(features_csv, labels_csv, output_file):
    # Import dfs
    labels = pd.read_csv(os.path.join(os.getcwd(), labels_csv))
    imp_feat = pd.read_csv(os.path.join(os.getcwd(), features_csv))

    # set label index
    labels.set_index('respondent_id', inplace=True)

    # IMPUTED 
    imp_feat.set_index('Unnamed: 0', inplace=True)
    imp_feat.sort_index(inplace=True)

    # merge_df options

    merged_df = imp_feat.join(labels)
    # merged_df = imp_feat_small.join(labels)

    df_h1n1 = merged_df.reset_index(drop=True).drop(['seasonal_vaccine'], axis=1)
    print(df_h1n1.shape)

    X = df_h1n1.iloc[:, :-1]
    y= df_h1n1.iloc[:,-1]

    X_train, X_val, y_train, y_val = train_test_split(X, y,
        test_size=0.1, stratify=y, random_state=42)
     
    # get feature names
    feature_names=list(X_train)

    # check shape
    print(X.shape)
    print(X_train.shape)

    # IMPUTED Scaling and 
    X_train = StandardScaler().fit_transform(X_train)
    print(X_train.shape)

    X_val = StandardScaler().fit_transform(X_val)

    clf =  SVC(kernel='rbf', probability=True).fit(X_train, y_train)
    X_train_summary = shap.kmeans(X_train, 10)
    explainer = shap.KernelExplainer(clf.predict_proba, X_train_summary)
    shap_values_train = explainer.shap_values(X_train)
    shap_values_test = explainer.shap_values(X_val)

    df_SVC = pd.DataFrame(shap_values_train[0].mean(0), index=X.columns, columns=['SVC']).sort_values('SVC', ascending=False)
    df_SVC.to_csv(output_file)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run shap with svm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('features_csv',
        default='Data/imputed_train_hot_encoded.csv', help='Path to features csv.')
    parser.add_argument('labels_csv',
        default='Data/training_set_labels.csv',help='Path to labels csv.')       
    parser.add_argument('output_file',
        default='SVC.csv', help='Path to save output file to.')
    args = parser.parse_args()

    run_shap(args.features_csv, args.labels_csv, args.output_file)
