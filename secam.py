import cv2
import pandas as pd
from ultralytics import YOLO
import easyocr
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model_yolo = YOLO("yolov8s.pt")
reader = easyocr.Reader(['en'])

def RGB(event, x, y, flags, param):
    colorsBGR = [x, y]
    print(colorsBGR)

cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)  # Add WINDOW_NORMAL flag
cv2.setMouseCallback('RGB', RGB)

video_path = r"C:\Users\Varshini\Downloads\yolov8-opencv-win11-main\yolov8-opencv-win11-main\amb.webm"
cap = cv2.VideoCapture(video_path)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)

# Add "ambulance" to the list of recognized classes
class_list.append("ambulance")

# Dictionary to track object labels using their unique IDs
object_labels = {}

count = 0
while True:
    try:
        ret, frame = cap.read()

        if not ret or frame is None:
            break

        count += 1
        if count % 3 != 0:
            continue

        results_yolo = model_yolo.predict(frame)
        a = results_yolo[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        confidence_threshold = 0.5
        px = px[px[4] > confidence_threshold]

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]

            region_of_interest = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            print(f"Class: {c}")
            text = pytesseract.image_to_string(thresh)

            # Check if the object is a vehicle and contains "AMBULANCE" in text
            if c in ["truck", "bus", "train", "car"]:
                if "AMBULANCE" in text:
                    # If the object is recognized as an ambulance, update the label
                    c = "ambulance"
                    # Assign a unique ID to the object if not already assigned
                    object_id = hash((x1, y1, x2, y2, c))
                    if object_id not in object_labels:
                        object_labels[object_id] = c

                    # Update the frame with the ambulance label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Ambulance", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # If not recognized as an ambulance, update the label using the object ID
                    object_id = hash((x1, y1, x2, y2, c))
                    if object_id in object_labels:
                        c = object_labels[object_id]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        # Convert from BGR to RGB before displaying
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show the video in the maximum RGB frame size
        cv2.namedWindow('RGB', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('RGB', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("RGB", frame_rgb)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

cap.release()
cv2.destroyAllWindows()
