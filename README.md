# 🐶 YOLOv4 Object Detection - Dog Detection

This project demonstrates **object detection using YOLOv4** on an image of a dog. It uses OpenCV’s deep learning module (`cv2.dnn`) to load YOLOv4 configuration, weights, and perform detection.

---

## 📁 Project Structure

```
YOLO4-REQUIREMENTS/
├── .gitignore
├── coco.names
├── detect.py
├── dog.png
├── yolov4.cfg
├── yolov4.weights
└── venv/
```

---

## 🚀 Requirements

- Python 3.x
- OpenCV
- NumPy

### 📦 Install dependencies:
```bash
pip install opencv-python numpy
```

---

## ⚙️ How it Works

- Loads YOLOv4 model (`yolov4.cfg` + `yolov4.weights`)
- Reads class names from `coco.names`
- Loads and processes the image (`dog.png`)
- Detects objects using the YOLOv4 network
- Draws bounding boxes and class labels on the image

---

## ▶️ Run the Script

```bash
python detect.py
```

---

## 📷 Output Screenshot

> You will see an output window showing detection results.  
> Below is a sample screenshot:

![Image](https://github.com/user-attachments/assets/598bc9f5-db5b-4498-ab01-859efdf88536)

> **Note**: Add your own screenshot by creating a folder `screenshots/` and saving the detection result as `output.png`.

---

## 🧾 .gitignore Highlights

```gitignore
venv/
*.weights
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.DS_Store
```

---

## 📚 Resources

- YOLOv4 Paper: https://arxiv.org/abs/2004.10934
- Official Darknet Repo: https://github.com/AlexeyAB/darknet

---

## 📌 Author

Made with ❤️ using Python and OpenCV.

