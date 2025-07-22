#from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
#results = model.train(data="config.yaml", epochs=1)  # train the model
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="config.yaml",
        epochs=30,
        batch=16,
        cache=True
    )

if __name__ == "__main__":
    main()

