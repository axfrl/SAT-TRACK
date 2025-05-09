import cv2
import numpy as np
import os
import sys
import traceback

# --- Paramètres du damier ---
CHECKERBOARD = (8, 6)  # coins intérieurs
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- Préparation des points 3D ---
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Listes pour stocker les points 3D et 2D
objpoints = []
imgpoints = []

# RTSP stream
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
rtsp_url = "rtsp://192.168.241.205:8554/head_color"

cap = None
try:
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError("Impossible d’ouvrir le flux RTSP")

    print("Appuie sur [espace] pour capturer une image, [c] pour calibrer, [q] pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Échec de lecture du flux")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if found:
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, found)

        cv2.imshow("Calibration View", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Espace : capturer l’image
            if found:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                print(f"[{len(objpoints)}] Image capturée")
            else:
                print("Damier non détecté")
        elif key == ord('c'):  # 'c' : lancer la calibration
            if len(objpoints) < 5:
                print("Capture au moins 5 images valides avant calibration.")
                continue
            print("Calibration en cours...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            print("Calibration terminée.")
            print("Matrice de la caméra :\n", mtx)
            print("Coefficients de distorsion :\n", dist)
        elif key == ord('q'):  # Quitter
            break

except Exception as e:
    print("Erreur lors du processus de calibration :")
    traceback.print_exc()

finally:
    if cap:
        cap.release()
    cv2.destroyAllWindows()
