import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix, classification_report
import numpy as np;
from sklearn.preprocessing import StandardScaler

def load_models():
    model_classification = pickle.load(open('model/model_classification.pkl', 'rb'))
    model_regression = pickle.load(open('model/model_regression.pkl', 'rb'))
    return model_classification, model_regression


def machine_learning():
    st.title("Datasets")
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if uploaded_file.name == "cart_transdata_filtered.csv":
            st.write("Файл cart_transdata_filtered.csv был загружен")
            y = data["fraud"]
            X = data.drop(["fraud"], axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
            scaler = StandardScaler()
            X_train_scaled_numeric = scaler.fit_transform(X_train.iloc[:, :3])
            X_test_scaled_numeric = scaler.transform(X_test.iloc[:, :3])
            X_train_binary = X_train.iloc[:, 3:7]
            X_test_binary = X_test.iloc[:, 3:7]
            X_train_scaled = np.column_stack((X_train_scaled_numeric, X_train_binary))
            X_test_scaled = np.column_stack((X_test_scaled_numeric, X_test_binary))
            model_classification = pickle.load(open('model/model_classification.pkl', 'rb'))
            predictions_classification = model_classification.predict(X_test_scaled)
            accuracy_classification = accuracy_score(y_test, predictions_classification)
            st.success(f"Точность: {accuracy_classification}")

        elif uploaded_file.name == "trip_duratopn_filtered.csv":
            st.write("Файл trip_duratopn_filtered.csv был загружен")
            
            
            model_regression = pickle.load(open('model/model_regression.pkl', 'rb'))
            y = data["trip_duration"]
            X = data.drop(["trip_duration"], axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            predictions_regression = model_regression.predict(X_test)
            r2_score_regression = r2_score(y_test, predictions_regression)
            st.success(f"Коэффициент детерминации (R²): {r2_score_regression}")


        else:
            st.write("Загружен файл неизвестного формата")


if __name__ == "__main__":
    machine_learning()