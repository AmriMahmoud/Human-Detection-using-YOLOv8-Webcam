 👤 Human Detection using YOLOv8 + Webcam

This project uses a custom-trained [YOLOv8](https://github.com/ultralytics/ultralytics) model to **detect humans in real-time** using your **laptop webcam**. It’s built with Python, OpenCV, and the Ultralytics YOLO library.

---


 🚀 Features

- ✅ Real-time human detection
- ✅ Runs directly from your laptop webcam
- ✅ YOLOv8 - fast, lightweight, accurate
- ✅ Easy to train and fine-tune on custom data
- ✅ GPU support for fast training and inference

---


## ⚡ Use GPU for Best Performance

> ✅ It is **strongly recommended** to use a **GPU** (via CUDA) for both training and real-time testing to achieve faster results.

If you're using a compatible NVIDIA GPU, ensure the following:

- Install CUDA and cuDNN
- Your environment has `torch` with GPU support:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## To check if YOLOv8 is using the GPU:

  import torch
  print(torch.cuda.is_available())  # Should return True


