from ultralytics import YOLO
import json
import os
import yaml
from PIL import Image
import shutil
import random

def rmDatasets():
    # Remove the datasets folder if it exists
    datasets_dir = "datasets"
    if os.path.exists(datasets_dir):
        shutil.rmtree(datasets_dir)
        print(f"{datasets_dir} folder removed successfully.")

def processInput():
    rmDatasets()

    # Filepaths
    json_file = r"input/tags.json"  # Path to your JSON file
    output_dir = r"datasets/labels"  # Directory to save YOLO annotation files
    data_yml_file = r"datasets/data.yml"  # Path to your data.yml file
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract unique classes from the JSON
    classes = set()
    for image_id, image_data in data["_via_img_metadata"].items():
        for region in image_data["regions"]:
            for attribute, value in region["region_attributes"].items():
                if value:  # If the attribute is marked as true
                    classes.add(attribute)

    # Convert the set of classes to a sorted list
    classes = sorted(classes)

    # Update the data.yml file
    data_yml = {
        "train": os.path.abspath(r"./datasets/images/train"),
        "val": os.path.abspath(r"./datasets/images/val"),
        "test": os.path.abspath(r"./datasets/images/test"),
        "nc": len(classes),  # Number of classes
        "names": classes     # List of class names
    }

    with open(data_yml_file, "w") as f:
        yaml.dump(data_yml, f, default_flow_style=False)

    print(f"data.yml updated successfully with {len(classes)} classes!")

    # Process each image in the JSON
    for image_id, image_data in data["_via_img_metadata"].items():
        filename = image_data["filename"]
        regions = image_data["regions"]

        # Get image dimensions
        image_path = os.path.join("input/images", filename)  # Adjust the path as needed
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Prepare the YOLO annotation file
        yolo_annotations = []
        for region in regions:
            shape = region["shape_attributes"]
            attributes = region["region_attributes"]

            # Extract bounding box coordinates
            x = shape["x"]
            y = shape["y"]
            width = shape["width"]
            height = shape["height"]

            # Calculate YOLO format values
            x_center = (x + width / 2) / image_width
            y_center = (y + height / 2) / image_height
            norm_width = width / image_width
            norm_height = height / image_height

            # Find the class ID
            for key, value in attributes.items():
                if value:  # If the attribute is marked as true
                    class_id = classes.index(key)  # Ensure `classes` is defined elsewhere
                    yolo_annotations.append(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}")

        # Save the annotations to a .txt file
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_filepath = os.path.join(output_dir, txt_filename)
        with open(txt_filepath, "w") as txt_file:
            txt_file.write("\n".join(yolo_annotations))

    print("YOLO annotations generated successfully!")

    # Paths
    image_dir = r"input/images"  # Path to your images folder
    label_dir = r"datasets/labels"  # Path to your labels folder
    output_dir = r"datasets/"  # Base output directory

    # Output subdirectories
    train_image_dir = os.path.join(output_dir, "images/train")
    val_image_dir = os.path.join(output_dir, "images/val")
    test_image_dir = os.path.join(output_dir, "images/test")
    train_label_dir = os.path.join(output_dir, "labels/train")
    val_label_dir = os.path.join(output_dir, "labels/val")
    test_label_dir = os.path.join(output_dir, "labels/test")

    # Create output directories
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    # Shuffle and split
    random.shuffle(image_files)
    train_split = int(0.7 * len(image_files))  # 70% for training
    val_split = int(0.85 * len(image_files))  # 15% for validation, 15% for testing

    train_files = image_files[:train_split]
    val_files = image_files[train_split:val_split]
    test_files = image_files[val_split:]

    # Function to move files
    def move_files(file_list, src_image_dir, src_label_dir, dest_image_dir, dest_label_dir):
        for file in file_list:
            # Move image file
            src_image_path = os.path.join(src_image_dir, file)
            dest_image_path = os.path.join(dest_image_dir, file)
            shutil.copy(src_image_path, dest_image_path)

            # Move corresponding label file
            label_file = os.path.splitext(file)[0] + ".txt"
            src_label_path = os.path.join(src_label_dir, label_file)
            dest_label_path = os.path.join(dest_label_dir, label_file)
            if os.path.exists(src_label_path):
                shutil.move(src_label_path, dest_label_path)

    # Move files to their respective folders
    move_files(train_files, image_dir, label_dir, train_image_dir, train_label_dir)
    move_files(val_files, image_dir, label_dir, val_image_dir, val_label_dir)
    move_files(test_files, image_dir, label_dir, test_image_dir, test_label_dir)

    print("datasets shuffled and split into train, val, and test folders!")

processInput()

# Load a pretrained YOLOv11 model
model = YOLO('models/yolo11n.pt')

# Train the model
model.train(
    epochs=100,             # Number of training epochs
    imgsz=640,              # Image size
    batch=16,               # Batch size (adjust based on GPU memory)
    device='cpu',               # GPU device (use 'cpu' for CPU training)
    workers=8,              # Number of data loader workers
    project='runs/train',   # Directory to save training results
    name='first_try',       # Name of the training run
    data='datasets/data.yml'         # Path to your datasets configuration file
)

rmDatasets()
