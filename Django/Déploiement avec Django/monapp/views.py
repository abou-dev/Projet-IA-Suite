from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import JsonResponse
import pickle
import pefile
import os
import pandas as pd

def predict(request):
    if request.method == 'POST':
        # Charger le modèle
        with open('E:\\IA_Test_TP1\\optimized_lightgbm_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        # Obtenir les données du formulaire
        file = request.FILES['executable_file']
        file_name = default_storage.save(file.name, file)
        file_path = default_storage.path(file_name)

        # Ici, vous devez utiliser un outil pour extraire les caractéristiques de l'exécutable
        # Pour l'exemple, je simule l'extraction de caractéristiques
        features = extract_features_from_executable(file_path)

        # Faire la prédiction
        prediction = int(model.predict([features])[0])
        # Retourner la prédiction en tant que réponse JSON
        if prediction == 0:
            prediction_text = "Programme Non Légitime"
        else:
            prediction_text = "Programme Légitime"

            # Passer le résultat à la page result.html
        return render(request, 'monapp/result.html', {'prediction_text': prediction_text})

    return render(request, 'monapp/predict.html')


def extract_features_from_executable(file_path):
    """
    Extrait des caractéristiques spécifiques d'un fichier PE (Portable Executable).

    :param file_path: Chemin vers le fichier exécutable à analyser.
    :return: Une liste contenant les valeurs des caractéristiques spécifiées.
    """
    try:
        pe = pefile.PE(file_path)

        # Initialisation d'un dictionnaire pour stocker les caractéristiques
        features = {}

        # Extraction des caractéristiques
        features['AddressOfEntryPoint'] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        features['MajorLinkerVersion'] = pe.OPTIONAL_HEADER.MajorLinkerVersion
        features['MajorImageVersion'] = pe.OPTIONAL_HEADER.MajorImageVersion
        features['MajorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
        features['DllCharacteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
        features['SizeOfStackReserve'] = pe.OPTIONAL_HEADER.SizeOfStackReserve
        features['NumberOfSections'] = len(pe.sections)

        # Pour la ResourceSize, vous devrez sommer la taille de toutes les ressources.
        # Ceci est un exemple qui suppose que la structure des ressources est standard et présente.
        # Veuillez adapter selon les spécificités de votre utilisation.
        resource_size = 0
        if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
            for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                if hasattr(resource_type, 'directory'):
                    for resource_id in resource_type.directory.entries:
                        if hasattr(resource_id, 'directory'):
                            for resource_lang in resource_id.directory.entries:
                                resource_size += resource_lang.data.struct.Size
        features['ResourceSize'] = resource_size

        # Retourner les caractéristiques sous forme de liste pour correspondre à l'ordre attendu
        return [features[feature_name] for feature_name in
                ['AddressOfEntryPoint', 'MajorLinkerVersion', 'MajorImageVersion', 'MajorOperatingSystemVersion',
                 'DllCharacteristics', 'SizeOfStackReserve', 'NumberOfSections', 'ResourceSize']]

    except pefile.PEFormatError as e:
        print(f"Erreur lors de l'analyse du fichier {file_path}: {str(e)}")
        # Retourner une liste de zéros si le fichier ne peut pas être analysé
        return [0] * 8  # Assurez-vous que cela correspond au nombre de caractéristiques attendues
def result(request):
    # Cette vue est actuellement vide, car nous passons simplement les données depuis la vue predict.
    # Si tu veux effectuer des actions spécifiques ici, tu peux les ajouter.
        return render(request, 'monapp/result.html')
def home(request):
    return render(request, 'monapp/home.html')
