import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time

#streamlit run my_app.py --server.enableXsrfProtection=false
#================================================================================================================================================================

# Titre principal de l'application
st.header(":blue[Présentation de détection de visages ]")
st.markdown("""
* **Présentation** : Cette application illustre les algorithmes utilisés pour la détection de visages en temps réel.
""")

# Sidebar pour la navigation
menu = st.sidebar.selectbox("Menu", ["Tableau de bord", "Détection de visages"])

if menu == "Tableau de bord":
    st.write(":green[Bienvenue sur l'application de détection de visages et d'objets.]")
    
elif menu == "Détection de visages":
    # Charger le modèle de cascade pour la détection de visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#================================================================================================================================================================

    # Fonction de détection de visages
    def detect_faces():
        # Initialiser la webcam
        cap = cv2.VideoCapture(0)
        captured_frame = None  # Variable pour stocker l'image capturée

        while True:
            # Lire les images de la webcam
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur de capture de la webcam")
                break

            # Convertir les images en niveaux de gris
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Détecter les visages
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Dessiner des rectangles autour des visages détectés
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Afficher les images avec les visages détectés
            cv2.imshow('Détection de Visages', frame)

            # Stocker l'image capturée
            captured_frame = frame.copy()

            # Sortir de la boucle si 'q' est pressé
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Libérer la webcam et fermer les fenêtres
        cap.release()
        cv2.destroyAllWindows()

        return captured_frame
#================================================================================================================================================================

    st.title("Détection de Visages en Temps Réel")
    st.write("Appuyez sur le bouton ci-dessous pour démarrer la détection de visages.")

    # Bouton pour démarrer la détection de visages
    if st.button("Démarrer la détection"):
        captured_image = detect_faces()  # Lancer la détection de visages et capturer l'image

        if captured_image is not None:
            st.image(captured_image, channels="BGR")
            st.success("Détection terminée. Vous pouvez capturer l'écran.")

            # Bouton pour capturer l'écran
            if st.button("Capturer l'écran"):
                # Créer un répertoire pour stocker les captures d'écran
                if not os.path.exists("captures"):
                    os.makedirs("captures")

                # Générer un nom de fichier unique
                file_name = f"captures/capture_{int(time.time())}.png"

                # Sauvegarder l'image capturée
                cv2.imwrite(file_name, captured_image)
                st.success(f"Capture d'écran enregistrée sous {file_name}")
#================================================================================================================================================================

# Footer
st.sidebar.text("© 2024 Mamadou MBOW - Machine Learning && Deep Learning ")
