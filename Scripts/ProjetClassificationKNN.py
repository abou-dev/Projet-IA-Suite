import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import time
from pathlib import Path



# Chargement du jeu de données
PROJECT_ROOT = Path(__file__).resolve().parents[1]
dataset_path = PROJECT_ROOT / 'Dataset' / 'DatasetmalwareExtrait.csv'
dataset = pd.read_csv(dataset_path)
X = dataset.drop('legitimate', axis=1)
y = dataset['legitimate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Entraînement et évaluation du modèle KNN par défaut
start_time = time.time()
default_model_knn = KNeighborsClassifier()
default_model_knn.fit(X_train, y_train)
default_time_knn = time.time() - start_time
predictions_default = default_model_knn.predict(X_test)
acc_default = accuracy_score(y_test, predictions_default)
f1_default = f1_score(y_test, predictions_default)
conf_matrix_default = confusion_matrix(y_test, predictions_default)

print("KNN - Modèle par défaut - Précision:", acc_default)
print("Score F1:", f1_default)
print("Matrice de confusion pour KNN - Modèle par défaut:\n", conf_matrix_default)
print("Temps d'entraînement pour le modèle par défaut:", default_time_knn, "secondes")

# Optimisation des hyperparamètres avec GridSearchCV pour KNeighborsClassifier
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance']
}

grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, n_jobs=-1)
start_time = time.time()
grid_search_knn.fit(X_train, y_train)
optimized_time_knn = time.time() - start_time

print("\nMeilleurs hyperparamètres pour KNN :", grid_search_knn.best_params_)
# Évaluation du modèle optimisé
optimized_model_knn = grid_search_knn.best_estimator_
predictions_optimized = optimized_model_knn.predict(X_test)
acc_optimized = accuracy_score(y_test, predictions_optimized)
f1_optimized = f1_score(y_test, predictions_optimized)
conf_matrix_optimized = confusion_matrix(y_test, predictions_optimized)


print("KNN - Modèle optimisé - Précision:", acc_optimized)
print("Score F1:", f1_optimized)
print("Matrice de confusion pour KNN - Modèle optimisé:\n", conf_matrix_optimized)
print(f"Temps d'entraînement pour l'optimisation KNN: {optimized_time_knn:.4f} secondes")
