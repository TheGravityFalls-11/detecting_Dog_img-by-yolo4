import cv2
import numpy as np

# Paths to files
config_path = "yolov4.cfg"
weights_path = "yolov4.weights"
classes_path = "coco.names"
image_path = "dog.png"  # 👈 Put your dog image here

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
classes = []
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load image
img = cv2.imread(image_path)
height, width = img.shape[:2]

# Prepare input
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Analyze results
conf_threshold = 0.5
nms_threshold = 0.4
boxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-max suppression to remove duplicates
indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw boxes
font = cv2.FONT_HERSHEY_SIMPLEX
for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, y - 10), font, 0.5, color, 2)

# Show result
cv2.imshow("YOLOv4 Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
