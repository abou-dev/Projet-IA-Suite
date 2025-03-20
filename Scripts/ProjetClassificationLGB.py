import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import lightgbm as lgb
import time

# Chargement du jeu de données
dataset_path = 'C:\\Users\\aboup\\PycharmProjects\\Face-recognition\\DatasetmalwareExtrait.csv'
dataset = pd.read_csv(dataset_path)
X = dataset.drop('legitimate', axis=1)
y = dataset['legitimate']

# Division du jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Définition du modèle LightGBM
lgbm = lgb.LGBMClassifier(random_state=42)

# Entraînement du modèle avec des hyperparamètres par défaut pour avoir une baseline
start_time = time.time()
lgbm.fit(X_train, y_train)
default_time = time.time() - start_time

# Évaluation du modèle par défaut
predictions = lgbm.predict(X_test)
print("LightGBM - Modèle par défaut - Précision:", accuracy_score(y_test, predictions))
print("Score F1:", f1_score(y_test, predictions))
print("Matrice de confusion:\n", confusion_matrix(y_test, predictions))
print("Temps d'entraînement pour le modèle par défaut:", default_time, "secondes")

# Optimisation des hyperparamètres avec GridSearchCV
param_grid = {
    'num_leaves': [31, 127],
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [30, 50, 100],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
}

grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
start_time = time.time()
grid_search.fit(X_train, y_train)
optimized_time = time.time() - start_time

# Affichage des meilleurs hyperparamètres
print("\nMeilleurs hyperparamètres pour LightGBM:", grid_search.best_params_)

# Évaluation du modèle optimisé
optimized_lgbm = grid_search.best_estimator_
predictions_optimized = optimized_lgbm.predict(X_test)
print("LightGBM - Modèle optimisé - Précision:", accuracy_score(y_test, predictions_optimized))
print("Score F1:", f1_score(y_test, predictions_optimized))
print("Matrice de confusion pour le modèle optimisé:\n", confusion_matrix(y_test, predictions_optimized))
print("Temps d'entraînement pour l'optimisation LightGBM:", optimized_time, "secondes")

model_filename = 'optimized_lightgbm_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(optimized_lgbm, file)

print(f"Le modèle optimisé a été sauvegardé sous {model_filename}")