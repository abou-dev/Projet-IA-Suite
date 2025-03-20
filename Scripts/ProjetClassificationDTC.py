import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import time

# Chargement du jeu de données
dataset_path = 'C:\\Users\\aboup\\PycharmProjects\\Face-recognition\\DatasetmalwareExtrait.csv'
dataset = pd.read_csv(dataset_path)
X = dataset.drop('legitimate', axis=1)
y = dataset['legitimate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Fonction pour évaluer un modèle et afficher ses performances
def evaluate_and_display_performance(model, X_test, y_test, model_name="Model"):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    print(f"{model_name} - Précision: {acc:.4f}, Score F1: {f1:.4f}")
    print(f"Matrice de confusion pour {model_name}:\n{conf_matrix}\n")

# Évaluation du modèle par défaut avec DecisionTreeClassifier
start_time = time.time()
default_model = DecisionTreeClassifier(random_state=42)
default_model.fit(X_train, y_train)
default_time = time.time() - start_time
evaluate_and_display_performance(default_model, X_test, y_test, "Decision Tree - Modèle par défaut")

# Optimisation des hyperparamètres avec GridSearchCV pour DecisionTreeClassifier
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

start_time = time.time()
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
optimized_time = time.time() - start_time

print("Meilleurs hyperparamètres pour Decision Tree :", grid_search.best_params_)

# Évaluation du modèle optimisé
optimized_model = grid_search.best_estimator_
evaluate_and_display_performance(optimized_model, X_test, y_test, "Decision Tree - Modèle optimisé")

# Affichage des temps d'entraînement
print(f"Temps d'entraînement avant optimisation: {default_time:.4f} secondes")
print(f"Temps d'entraînement après optimisation: {optimized_time:.4f} secondes")
