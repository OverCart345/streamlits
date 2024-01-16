import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.cluster import KMeans
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.cluster  import KMeans
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

cart_data = pd.read_csv('cart_transdata_filtered.csv')
X = cart_data.drop('fraud', axis=1)
y = cart_data['fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

parameters = {'n_neighbors': [2,3,5,10], 'weights': ['uniform', 'distance']}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, parameters, cv=5)
clf.fit(X_train, y_train)

best_params = clf.best_params_

best_knn = KNeighborsClassifier(**best_params)
best_knn.fit(X_train, y_train)

with open('best_knn_model.pkl', 'wb') as file:
    pickle.dump(best_knn, file)


best_gmm_model = KMeans(n_clusters=5)
best_gmm_model.fit(X_train)

with open('best_kmeans5_model.pkl', 'wb') as file:
    pickle.dump(best_gmm_model, file)


params = {'depth': [4, 6, 8], 'learning_rate': [0.01, 0.05, 0.1], 'iterations': [500]}
cat_grid = GridSearchCV(CatBoostClassifier(), params, cv=3)

catboost = CatBoostClassifier(depth=4, learning_rate=0.05, iterations=500)
catboost.fit(X_train, y_train)

with open('best_catboost_model.pkl', 'wb') as file:
    pickle.dump(catboost, file)


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
}

rfc = RandomForestClassifier(random_state=42)
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, verbose=10)
CV_rfc.fit(X_train, y_train)

best_rfc = RandomForestClassifier(**CV_rfc.best_params_)
best_rfc.fit(X_train, y_train)

with open('best_RandomTreeClassifer_model.pkl', 'wb') as file:
    pickle.dump(best_rfc, file)


base_classifiers = [
    ('logic', LogisticRegression()),
    ('KNN', KNeighborsClassifier()),
    ('dtc',DecisionTreeClassifier())
]
meta_classifier = LogisticRegression(random_state=42)

stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)
stacking_classifier.fit(X_train, y_train)

with open('best_Stacking_model.pkl', 'wb') as file:
    pickle.dump(stacking_classifier, file)


model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=10)

model.save("D:/Cpython/streamlit-rgr-main/bestmodels/" + 'sdf.h5')