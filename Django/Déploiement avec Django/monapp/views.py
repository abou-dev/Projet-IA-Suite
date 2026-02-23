from django.core.files.storage import default_storage
from django.shortcuts import render
import pickle
import pefile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / 'Models' / 'optimized_lightgbm_model.pkl'
ALLOWED_EXTENSIONS = {'.exe', '.dll'}
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB

def predict(request):
    if request.method == 'POST':
        file = request.FILES.get('executable_file')
        if not file:
            return render(request, 'monapp/predict.html', {'error_message': "Veuillez sélectionner un fichier."})

        file_extension = Path(file.name).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return render(
                request,
                'monapp/predict.html',
                {'error_message': "Format invalide. Utilisez un fichier .exe ou .dll."},
            )

        if file.size > MAX_UPLOAD_SIZE:
            return render(
                request,
                'monapp/predict.html',
                {'error_message': "Fichier trop volumineux. Taille maximale: 20 MB."},
            )

        try:
            with open(MODEL_PATH, 'rb') as model_file:
                model = pickle.load(model_file)
        except FileNotFoundError:
            return render(
                request,
                'monapp/predict.html',
                {'error_message': "Modèle introuvable. Vérifiez le dossier Models."},
            )

        file_name = default_storage.save(file.name, file)
        file_path = default_storage.path(file_name)

        features = extract_features_from_executable(file_path)

        prediction = int(model.predict([features])[0])
        if prediction == 0:
            prediction_text = "Programme Non Légitime"
        else:
            prediction_text = "Programme Légitime"

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
    return render(request, 'monapp/result.html')


def home(request):
    return render(request, 'monapp/home.html')
