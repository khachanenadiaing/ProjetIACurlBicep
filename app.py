from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import mediapipe as mp
import numpy as np
from io import BytesIO
from PIL import Image


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Gérer le téléchargement de la vidéo
        pass
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    # Récupérer le fichier de la vidéo téléchargée
    video_file = request.files['video']
    app.config['UPLOAD_FOLDER'] = 'uploads'
    # Enregistrer la vidéo dans le dossier 'uploads/'
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
    video_file.save(video_path)

    # Traiter la vidéo
    output_video_path = process_video_with_mediapipe(video_path)

    # Renvoyer le fichier de la vidéo traitée
    return send_file(output_video_path, as_attachment=True)

def process_video_with_mediapipe(video_path):
    # Implémentez ici le code de traitement de la vidéo avec MediaPipe
    # Initialisation des modules MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Charger la vidéo enregistrée
    cap = cv2.VideoCapture(video_path)
    # Initialisation des variables pour le comptage des flexions du bras
    counter = 0
    stage = None
    # Initialisation de l'instance MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Paramètres de sortie pour la vidéo traitée
        output_video_path = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = cap.get(cv2.CAP_PROP_FPS)
        out_video = cv2.VideoWriter(output_video_path, fourcc, out_fps, (out_width, out_height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Conversion de l'image en RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
        # Détection des poses
            results = pose.process(image)
        
        # Conversion de l'image en BGR pour l'affichage
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extraction des landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                # Coordonnées des articulations
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
                # Calcul de l'angle
                angle = calculate_angle(shoulder, elbow, wrist)
                            # Affichage de l'angle
                cv2.putText(image, str(angle), 
                        tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
                # Logique de comptage des flexions
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = "up"
                    counter += 1
                    print(counter)
            except:
                pass
            # Affichage des compteurs et du stade
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            # Rendu des détections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            # Convertir l'image pour l'afficher dans Jupyter
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(image)
            b = BytesIO()
            pil_im.save(b, format='jpeg')
            im_data = b.getvalue()
             # Affichage de la vidéo traitée
            #clear_output(wait=True)
            #display(Image(data=im_data))
        
            #Écrire l'image traitée dans la vidéo de sortie
            out_video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    # Enregistrer la vidéo traitée
    
    return output_video_path

# Fonction pour calculer l'angle entre trois points
def calculate_angle(a, b, c):
    a = np.array(a) # Premier point
    b = np.array(b) # Point intermédiaire
    c = np.array(c) # Dernier point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

if __name__ == '__main__':
    app.run(debug=True)