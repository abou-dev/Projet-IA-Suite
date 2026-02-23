
# Projet IA: Optimisation des Hyperparametres et Deploiement Django

## Description
Ce projet couvre:
- Classification de logiciels malveillants avec plusieurs modeles ML.
- Application web Django pour lancer la prediction depuis une interface.

## Prerequis
- Python 3.10+ (3.11 recommande)
- `pip`

## Installation rapide (portable)
Depuis la racine du depot:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

Les scripts Python et la vue Django utilisent des chemins relatifs base sur la structure du projet (pas de chemins absolus machine-specifiques).

## Lancer un script ML
```bash
python Scripts/ProjetClassificationDTC.py
```

## Lancer Django
```bash
cd "Django/Déploiement avec Django"
python manage.py runserver
```

## Variables d'environnement Django
`djangoProjectIATest1/settings.py` lit:
- `DJANGO_SECRET_KEY` (ou `SECRET_KEY`)
- `DJANGO_DEBUG` (par defaut: `True`)
- `DJANGO_ALLOWED_HOSTS` (ou `ALLOWED_HOSTS`, CSV)

Exemple Linux/macOS:
```bash
export DJANGO_SECRET_KEY="change-me-in-production"
export DJANGO_DEBUG="False"
export DJANGO_ALLOWED_HOSTS="127.0.0.1,localhost,example.com"
```

Exemple Windows PowerShell:
```powershell
$env:DJANGO_SECRET_KEY="change-me-in-production"
$env:DJANGO_DEBUG="False"
$env:DJANGO_ALLOWED_HOSTS="127.0.0.1,localhost,example.com"
```
