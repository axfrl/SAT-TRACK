import os
import cv2
import sys
import traceback

# Force TCP for FFMPEG via environment variable
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

print("OpenCV version:", cv2.__version__)
print("FFMPEG? ", cv2.getBuildInformation() )


# RTSP stream URL (standard format)
rtsp_url = "rtsp://192.168.241.205:8554/head_color"

cap = None

try:
    # Attempt to open the stream
    cap = cv2.VideoCapture(rtsp_url)

    print(f"Backend used: {cap.getBackendName()}")

    if not cap.isOpened():
        raise RuntimeError("Failed to open RTSP stream")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame")
            break

        cv2.imshow("RTSP Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("An error occurred while opening or reading the RTSP stream:")
    traceback.print_exc()
    sys.exit(1)

finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()