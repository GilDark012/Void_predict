from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import requests
import shutil
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from werkzeug.utils import secure_filename
import warnings
from ultralytics import YOLO
# from download_model import download_sam

# Initialisation de Flask
app = Flask(
    __name__,
    template_folder='templates',  # Chemin des fichiers HTML
    static_folder='static'       # Chemin des fichiers statiques
)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#Download model
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
# Charger le modèle SAM
MODEL_TYPE = "vit_b"
MODEL_PATH = os.path.join('models', 'sam_vit_b_01ec64.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Chargement du modèle SAM...")
try:
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
except TypeError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")

# Initialiser et charger le modèle
sam = sam_model_registry[MODEL_TYPE]()
sam.load_state_dict(state_dict, strict=False)
sam.to(device=device)
predictor = SamPredictor(sam)
print("Modèle SAM chargé avec succès!")

print("Chargement du modèle YOLO...")
yolo_model = YOLO('./models/best.pt')  # Remplace par le chemin de ton modèle YOLO si nécessaire
yolo_model.to(device)
# yolo_model.train(data='data.yaml', epochs=10, imgsz=640)
print("Modèle YOLO chargé avec succès!")


# Fonction pour générer une couleur unique pour chaque classe
def get_color_for_class(class_name):
    np.random.seed(hash(class_name) % (2**32))
    return tuple(np.random.randint(0, 256, size=3).tolist())

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or not file.filename:
            return "Aucun fichier sélectionné", 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return render_template('index.html', uploaded_image=filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/segment', methods=['POST'])
def segment():
    data = request.get_json()
    image_name = data.get('image_name')
    points = data.get('points')

    if not image_name or not points:
        return jsonify({'success': False, 'error': 'Données manquantes'}), 400

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    if not os.path.exists(image_path):
        return jsonify({'success': False, 'error': 'Image non trouvée'}), 404

    # Charger l'image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    annotated_image = image.copy()

    height, width, _ = image.shape
    masks_list = []
    class_id = 0  # Classe par défaut pour YOLO

    # Effectuer la segmentation avec SAM
    for point in points:
        x, y = point['x'], point['y']
        class_name = point.get('class', 'Unknown')
        color = get_color_for_class(class_name)

        masks, _, _ = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=False
        )
    

        # Annoter l'image
        annotated_image[masks[0] > 0] = color
        cv2.putText(annotated_image, class_name, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        masks_list.append(masks[0])
        

    # Sauvegarder l'image annotée
    annotated_filename = f"annotated_{image_name}"
    annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
    cv2.imwrite(annotated_path, annotated_image)

    # Sauvegarder les annotations au format YOLO
    annotation_filename = os.path.splitext(image_name)[0] + ".txt"
    annotation_path = os.path.join(app.config['UPLOAD_FOLDER'], annotation_filename)
    save_yolo_annotation(masks_list, class_id, image.shape, annotation_path)
    save_yolo_dataset(image_path, annotation_path)

    return jsonify({
        'success': True,
        'annotated_image': f"uploads/{annotated_filename}",
        'annotation_file': f"uploads/{annotation_filename}"
    })


def save_yolo_annotation(masks, class_id, image_shape, annotation_path):
    """Convert masks to YOLO format and save them as a .txt file."""
    height, width, _ = image_shape
    annotations = []

    # Find bounding box for each mask
    for mask in masks:
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue

        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # Convert bounding box to YOLO format (x_center, y_center, width, height)
        x_center = (x_min + x_max) / 2 / width
        y_center = (y_min + y_max) / 2 / height
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height

        annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Save annotations to a .txt file
    with open(annotation_path, 'w') as f:
        f.write("\n".join(annotations))

def save_yolo_dataset(image_path, annotation_path, dataset_type='train'):
    dataset_dir = os.path.join('datasets', dataset_type)
    os.makedirs(os.path.join(dataset_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels'), exist_ok=True)

    # Copier l'image et l'annotation dans les répertoires
    image_filename = os.path.basename(image_path)
    annotation_filename = os.path.basename(annotation_path)

    new_image_path = os.path.join(dataset_dir, 'images', image_filename)
    new_annotation_path = os.path.join(dataset_dir, 'labels', annotation_filename)

    # Utiliser shutil.move() pour gérer les conflits de fichiers
    shutil.move(image_path, new_image_path)
    shutil.move(annotation_path, new_annotation_path)

@app.route('/detect', methods=['POST'])
def detect_objects():
    file = request.files.get('image')
    if not file or not file.filename:
        return jsonify({'success': False, 'error': 'Aucun fichier sélectionné'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Charger l'image avec OpenCV
    image = cv2.imread(filepath)

    # Effectuer la détection d'objets avec YOLO
    results = yolo_model.predict(image)
    detections = results[0].boxes

    annotated_image = image.copy()

    # Boucle sur chaque détection
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
        confidence = detection.conf[0]
        class_id = int(detection.cls[0])
        class_name = yolo_model.names[class_id]

        # Dessiner le rectangle autour de l'objet
        color = get_color_for_class(class_name)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Sauvegarder et renvoyer l'image annotée
    detected_filename = f"detected_{filename}"
    detected_path = os.path.join(app.config['UPLOAD_FOLDER'], detected_filename)
    cv2.imwrite(detected_path, annotated_image)
    
    return jsonify({'success': True, 'detected_image': f"uploads/{detected_filename}"})

@app.route('/retrain', methods=['POST'])
def retrain_yolo():
    try:
        # Charger le modèle YOLO pré-entraîné
        model = yolo_model

        # Spécifie le chemin du dataset pour l'entraînement
        data_yaml = 'data.yaml'  # Assure-toi que ce fichier existe et est bien configuré

        # Lancer l'entraînement
        print("Démarrage de l'entraînement YOLO...")
        model.train(data=data_yaml, epochs=10, imgsz=640)

        return jsonify({'success': True, 'message': 'Modèle réentraîné avec succès'})
    except Exception as e:
        print(f"Erreur lors du réentraînement : {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3000)
