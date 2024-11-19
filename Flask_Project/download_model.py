import os
import requests

# Créer le dossier cible s'il n'existe pas
os.makedirs("models", exist_ok=True)

# URL du modèle et chemin de destination
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
output_path = "models/sam_vit_b_01ec64.pth"

# Téléchargement
print("Téléchargement du modèle...")
response = requests.get(url, stream=True)
with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("Téléchargement terminé ! Modèle sauvegardé dans :", output_path)