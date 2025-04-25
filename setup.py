import os

# Create the input folder structure
input_dir = "input"
images_dir = os.path.join(input_dir, "images")
tags_file = os.path.join(input_dir, "tags.json")

# Create directories if they don't exist
os.makedirs(images_dir, exist_ok=True)

# Create an empty tags.json
with open(tags_file, "w") as f:
    pass

print(f"Input folder structure created with an empty {tags_file} and subfolder {images_dir}.")