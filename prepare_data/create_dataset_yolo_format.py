import os
import pandas as pd

#  Your real base directory
base_dir = 'C:/Users/ASUS/Desktop/ML/human_detection_model/archive'
splits = ['train', 'valid', 'test']

class_map = {'person': 0}

for split in splits:
    # Adjusted for extra folder inside each split
    csv_path = os.path.join(base_dir, split, split, '_annotations.csv')
    df = pd.read_csv(csv_path)
    
    label_dir = os.path.join(base_dir, split, split, 'labels')
    os.makedirs(label_dir, exist_ok=True)
    
    for index, row in df.iterrows():
        filename = row['filename']
        image_w = row['width']
        image_h = row['height']
        class_name = row['class']
        
        x_min = row['xmin']
        y_min = row['ymin']
        x_max = row['xmax']
        y_max = row['ymax']
        
        x_center = ((x_min + x_max) / 2) / image_w
        y_center = ((y_min + y_max) / 2) / image_h
        width = (x_max - x_min) / image_w
        height = (y_max - y_min) / image_h
        
        class_id = class_map[class_name]
        
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))
        with open(label_path, 'a') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print(" Done converting CSV annotations to YOLO format.")
