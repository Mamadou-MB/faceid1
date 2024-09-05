import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

# Fonction pour charger les noms des classes
def load_classes(file):
    with open(file, "r") as f:
        return [line.strip() for line in f.readlines()]

# Fonction pour charger le modèle YOLOv4
def load_yolo_model(cfg_file, weights_file):
    return cv2.dnn.readNet(cfg_file, weights_file)

# Fonction pour effectuer la détection d'objets
def detect_objects(image, net, output_layers):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialiser les listes pour les boîtes englobantes, les confidences et les classes détectées
    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Boîte englobante
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

# Fonction pour dessiner les boîtes englobantes sur l'image
def draw_labels(image, boxes, confidences, class_ids, classes):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Couleur des boîtes englobantes (vert)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Fonction pour détecter les objets à partir de la webcam
def detect_objects_from_webcam(net, output_layers, classes):
    # Initialiser la webcam
    cap = cv2.VideoCapture(0)  # 0 pour la webcam par défaut

    while True:
        # Lire l'image de la webcam
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de l'accès à la webcam.")
            break

        # Effectuer la détection d'objets
        boxes, confidences, class_ids = detect_objects(frame, net, output_layers)

        # Dessiner les boîtes englobantes sur l'image de la webcam
        frame_with_boxes = draw_labels(frame.copy(), boxes, confidences, class_ids, classes)

        # Afficher le flux vidéo avec les objets détectés
        cv2.imshow("Webcam YOLOv4 Detection", frame_with_boxes)

        # Appuyer sur 'q' pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

# Charger le modèle YOLOv4 et les classes
st.title("🔍 Détection d'Objets et visages ")

config_path = "path/to/yolov4-tiny.cfg"
weights_path = "path/to/yolov4-tiny.weights"
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
names_file = "coco .names"

try:
    net = load_yolo_model(config_path, weights_path)
    classes = load_classes(names_file)
    output_layers = net.getUnconnectedOutLayersNames()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {str(e)}")
    st.stop()

# Charger l'image depuis l'interface Streamlit
st.sidebar.header("📂 Options")
uploaded_file = st.sidebar.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire l'image
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))
    
    # Effectuer la détection d'objets
    boxes, confidences, class_ids = detect_objects(img, net, output_layers)
    
    # Dessiner les boîtes englobantes
    img_with_boxes = draw_labels(img.copy(), boxes, confidences, class_ids, classes)
    
    # Afficher l'image originale et l'image annotée
    st.image(image, caption="Image originale", use_column_width=True, channels="RGB")
    st.image(img_with_boxes, caption="Image avec détection", use_column_width=True, channels="RGB")
    
    # Convertir l'image annotée en binaire pour le téléchargement
    buffer = io.BytesIO()
    Image.fromarray(img_with_boxes).save(buffer, format="PNG")
    buffer.seek(0)

    # Option pour télécharger l'image annotée
    st.sidebar.download_button(
        label="Télécharger l'image avec détection",
        data=buffer,
        file_name="annotated_image.png",
        mime="image/png"
    )

# Ajouter une option pour activer la détection via la webcam
if st.sidebar.button("Activer la détection via Webcam"):
    st.info("Appuyez sur 'q' pour quitter la détection via la webcam.")
    detect_objects_from_webcam(net, output_layers, classes)

# Footer
st.sidebar.text("© 2024 Mamadou MBOW - Machine Learning && Deep Learning")
