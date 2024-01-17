import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, rand_score
import pickle
from tensorflow.keras.models import load_model

# Константы
MODEL_SAVE_PATH = 'bestmodels/'
DATA_FILE_PATH = 'cart_transdata_filtered.csv'
DEVELOPER_PHOTO_PATH = 'full.jpg'
SIDEBAR_IMAGE_PATH = 'circle.png'

# Загрузка и подготовка данных
data = pd.read_csv(DATA_FILE_PATH)
X = data.drop('fraud', axis=1)
y = data['fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Функции визуализации
def plot_comparison_of_chip_and_pin(data):
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='used_chip', hue='used_pin_number', ax=ax)
    ax.set_title('Сравнение использования чипа и PIN-кода в транзакциях')
    ax.set_xlabel('Использование чипа')
    ax.set_ylabel('Количество транзакций')
    st.pyplot(fig)

def plot_correlation_matrix(data):
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Корреляционная матрица')
    st.pyplot(fig)

def plot_correlation_matrix2(data):
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='distance_from_home', y='distance_from_last_transaction', hue='fraud', ax=ax)
    ax.set_title('Взаимосвязь между расстоянием от дома и последней транзакцией')
    ax.set_xlabel('Расстояние от дома')
    ax.set_ylabel('Расстояние от последней транзакции')
    st.pyplot(fig)

def plot_correlation_matrix3(data):
    fig, ax = plt.subplots()
    sns.countplot(x='fraud', data=data, ax=ax)
    ax.set_title('Сравнение количества мошеннических и немошеннических транзакций')
    st.pyplot(fig)

def plot_correlation_matrix4(data):
    fig, ax = plt.subplots()
    sns.boxplot(x='fraud', y='distance_from_home', data=data, ax=ax)
    ax.set_title('Сравнение расстояний от дома в мошеннических и немошеннических транзакциях')
    st.pyplot(fig)


# Функции страниц
def page_developer_info():
    st.title("Информация о разработчике")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Контактная информация")
        st.write("**ФИО:** Чепурко Артём Иванович")
        st.write("**Номер учебной группы:** ФИТ-222")
    with col2:
        st.subheader("Фотография")
        st.image(DEVELOPER_PHOTO_PATH, width=250)
    st.markdown("---")
    st.header("Тема РГР")
    st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")

def page_dataset_info():
    st.title("Информация о наборе данных")
    st.markdown("""   
# Название датасета
Описание датасета 'card_transdata.csv'

## Описание датасета
Этот датасет содержит информацию о транзакциях карт. Данные могут быть использованы для анализа покупательского поведения, а также для выявления мошеннических операций.

## Описание столбцов
- **distance_from_home**: Расстояние от дома до места совершения транзакции (вещественное число).
- **distance_from_last_transaction**: Расстояние от последней транзакции (вещественное число).
- **ratio_to_median_purchase_price**: Отношение суммы транзакции к медианной цене покупки (вещественное число).
- **repeat_retailer**: Повторная покупка у одного и того же продавца (целое число).
- **used_chip**: Использование чипа при транзакции (целое число).
- **used_pin_number**: Использование PIN-кода при транзакции (целое число).
- **online_order**: Онлайн заказ (целое число).
- **fraud**: Мошенническая транзакция (целое число).

## Особенности предобработки
- Преобразование типов данных для 'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order' и 'fraud' в целочисленный формат.
- Определение и удаление выбросов для 'distance_from_home' и 'distance_from_last_transaction' с использованием IQR-оценок.
- Применение undersampling для балансировки классов в столбце 'fraud'.""")

def page_data_visualization():
    st.title("Визуализации данных транзакций")
    plot_comparison_of_chip_and_pin(data)
    plot_correlation_matrix(data)
    plot_correlation_matrix2(data)
    plot_correlation_matrix3(data)
    plot_correlation_matrix4(data)

def page_ml_prediction():
    st.title("Предсказания моделей машинного обучения")

    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")

    if uploaded_file is None:
        st.subheader("Введите данные для предсказания:")

        input_data = {}
        feature_names = {
            'float': {'Расстояние от дома (км)': 'distance_from_home', 
                  'Расстояние от последней транзакции (км)': 'distance_from_last_transaction', 
                  'Отношение к медианной цене покупки': 'ratio_to_median_purchase_price'},
            'int': {'Повторный заказ': 'repeat_retailer', 
                'Использование чипа': 'used_chip', 
                'Использование_пин-кода': 'used_pin_number', 
                'Онлайн_заказ': 'online_order'}
        }

        for dtype, features in feature_names.items():
            for feature_rus, feature_eng in features.items():
                if dtype == 'float':
                    input_data[feature_eng] = st.number_input(f"{feature_rus}", min_value=0.0, max_value=100000.0, value=0.0)
                else:  # dtype == 'int'
                    selected_option = st.selectbox(f"{feature_rus}", ['да', 'нет'], index=1)
                    input_data[feature_eng] = 1 if selected_option == 'да' else 0
        if st.button('Сделать предсказание'):
            model_catboost, model_kmeans, model_knn, model_random_tree, model_stacking, model_neiro = deserialisation()


            input_df = pd.DataFrame([input_data])

            prediction_catboost = model_catboost.predict(input_df)
            prediction_knn = model_knn.predict(input_df)
            prediction_random_tree = model_random_tree.predict(input_df)
            prediction_stacking = model_stacking.predict(input_df)
            prediction_neiro = (model_neiro.predict(input_df) > 0.5).astype(int)

            st.success(f"Результат предсказания CatBoost: {prediction_catboost[0]}")
            st.success(f"Результат предсказания KNN: {prediction_knn[0]}")
            st.success(f"Результат предсказания RandomTreeClassifier: {prediction_random_tree[0]}")
            st.success(f"Результат предсказания Stacking: {prediction_stacking[0]}")
            st.success(f"Результат предсказания нейронной сети Tensorflow: {prediction_neiro[0]}")
    else:
        try:
            model_catboost, model_kmeans, model_knn, model_random_tree, model_stacking, model_neiro = deserialisation()

            prediction_catboost = model_catboost.predict(X_test)
            prediction_kmeans = model_kmeans.predict(X_test)
            prediction_knn = model_knn.predict(X_test)
            prediction_random_tree = model_random_tree.predict(X_test)
            prediction_stacking = model_stacking.predict(X_test)
            prediction_neiro = model_neiro.predict(X_test).round()

            rand_score_ml2 = rand_score(y_test, prediction_kmeans)
            accuracy_Knn = accuracy_score(y_test, prediction_knn)
            accuracy_catboost = accuracy_score(y_test, prediction_catboost)
            accuracy_random_tree = accuracy_score(y_test, prediction_random_tree)
            accuracy_stacking = accuracy_score(y_test, prediction_stacking)
            accuracy_neiro = accuracy_score(y_test, prediction_neiro)

            plt.figure(figsize=(8, 6))
            plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=prediction_kmeans, cmap='viridis', marker='o')
            plt.title('Распределение по кластерам KMeans')
            plt.xlabel('Признак 1')
            plt.ylabel('Признак 2')
            plt.colorbar()
            st.pyplot(plt)

            st.success(f"rand_score KMeans: {rand_score_ml2}")
            st.success(f"Точность KNeighborsClassifier: {accuracy_Knn}")
            st.success(f"Точность CatBoostClassifier: {accuracy_catboost}")
            st.success(f"Точность RandomForestClassifier: {accuracy_random_tree}")
            st.success(f"Точность StackingClassifier: {accuracy_stacking}")
            st.success(f"Точность нейронной сети: {accuracy_neiro}")

        except Exception as e:
            st.error(f"Произошла ошибка при обработке файла: {e}")

# Функция для загрузки моделей
def deserialisation():
    model_names = ['best_catboost_model.pkl', 'best_kmeans_model.pkl', 'best_knn_model.pkl',
                    'best_RandomTreeClassifer_model.pkl', 'best_Stacking_model.pkl', 'best_neiro_model.h5']
    models = []
    for model_name in model_names:
        model_path = MODEL_SAVE_PATH + model_name
        if model_name.endswith('.h5'):
            model = load_model(model_path)
        else:
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
        models.append(model)
    return models

# Основной код Streamlit
def main():
    st.markdown("""
<style>
    body {
        background-color: #f9f9f9;
        color: #333333;
    }
    .sidebar .sidebar-content {
        background-color: #4287f5;
    }
</style>
""", unsafe_allow_html=True)
    st.sidebar.image(SIDEBAR_IMAGE_PATH, width=100)
    page = st.sidebar.radio("Выберите страницу:", ["Информация о разработчике", "Информация о наборе данных", "Визуализации данных", "Предсказание модели ML"])
    
    if page == "Информация о разработчике":
        page_developer_info()
    elif page == "Информация о наборе данных":
        page_dataset_info()
    elif page == "Визуализации данных":
        page_data_visualization()
    elif page == "Предсказание модели ML":
        page_ml_prediction()

if __name__ == "__main__":
    main()