import cv2
import numpy as np
import os

# Chemins des fichiers
input_path = "/home/alphafalcon/Videos/Webcam/2025-04-22-102544.webm"
output_path = "/home/alphafalcon/Videos/input/2025-04-22-102544.mp4"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Fonction pour recadrer le FOV (85° à 60°)
def crop_fov(image, crop_ratio=0.63):
    h, w = image.shape[:2]
    new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    cropped_image = image[start_y:start_y + new_h, start_x:start_x + new_w]
    return cropped_image

# Ouvrir le fichier vidéo .webm
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"Erreur : Impossible d'ouvrir le fichier {input_path}")
    exit()

# Obtenir les propriétés de la vidéo
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 3840
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 1080

# Définir la résolution cible pour le modèle (par exemple, 1280x720)
target_resolution = (1280, 720)

# Configurer l'écrivain vidéo pour .mp4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec pour .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, target_resolution)
if not out.isOpened():
    print(f"Erreur : Impossible de créer le fichier {output_path}")
    cap.release()
    exit()

# Boucle de traitement du flux vidéo
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Étape 1 : Séparer les flux binoculaires
    left_image = frame[:, :width//2]  # 1920x1080
    right_image = frame[:, width//2:]  # 1920x1080

    # Étape 2 : Ajuster le FOV (85° à 60°)
    left_cropped = crop_fov(left_image)

    # Étape 3 : Redimensionner à la résolution cible
    left_resized = cv2.resize(left_cropped, target_resolution, interpolation=cv2.INTER_LINEAR)

    # Étape 4 : Convertir en RGB (facultatif, si le modèle l'exige)
    # Note : Pour l'enregistrement, on garde BGR pour compatibilité avec VideoWriter
    # Si le modèle a besoin de RGB, vous pouvez convertir ici et reconvertir pour l'enregistrement
    # left_rgb = cv2.cvtColor(left_resized, cv2.COLOR_BGR2RGB) / 255.0

    # Écrire l'image prétraitée dans le fichier .mp4
    out.write(left_resized)

    # Afficher pour vérification (facultatif)
    cv2.imshow("Processed Left", left_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Fichier enregistré avec succès : {output_path}")