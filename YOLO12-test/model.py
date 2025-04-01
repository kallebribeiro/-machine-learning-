import cv2
import mss
from ultralytics import YOLO
import numpy as np

# Load a COCO-pretrained YOLO12n model
model = YOLO("")

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLO12n model on the 'bus.jpg' image
# results = model("YOLO12-test/teste")

# Initialize video capture using the camera
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# follow = False

# while True:
#     success, img = cap.read()
#     
#     if success:
#         if follow:
#             # Perform tracking with the model if 'follow' is True
#             results = model.track(img, persist=True)
#         else:
#             # Perform object detection with the YOLO model
#             results = model(img)
#         
#         # For each result, plot on the image
#         for result in results:
#             img = result.plot()
#         
#         # Display the image on the screen
#         cv2.imshow("Screen", img)
#     
#     # Check pressed key
#     k = cv2.waitKey(1)
#     if k == ord('q'):
#         break

# # Release video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()
# print('Shutting down :)')

# Run inference with the YOLO12n model on the 'bus.jpg' image
# results = model("YOLO12-test/teste")

# Define the screen area for capture (adjust as needed)
with mss.mss() as sct:
    monitor = sct.monitors[0]  # Capture the entire screen

# Create a screen capture object
with mss.mss() as sct:
    monitor = sct.monitors[0]  # Capture the entire screen

    while True:
        # Capture the screen
        screenshot = sct.grab(monitor)

        # Convert to a format usable by OpenCV
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Perform prediction with the YOLO model
        results = model.predict(frame, conf=0.5)  # Adjust confidence as needed

        # Add annotations on the screen
        for r in results:
            annotated_frame = r.plot()

        # Display the screen with detections
        cv2.imshow("Real-Time Detection on PC Screen", annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close windows
cv2.destroyAllWindows()