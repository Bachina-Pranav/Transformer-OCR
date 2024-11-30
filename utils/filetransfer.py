import os
import random
import shutil

def move_images(source_dir, dest_dir, num_images):
    # Get a list of all PNG files in the source directory
    png_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    
    # Sample 5000 images randomly
    sampled_images = random.sample(png_files, min(num_images, len(png_files)))

    # Move sampled images to the destination directory
    for image in sampled_images:
        src_path = os.path.join(source_dir, image)
        dest_path = os.path.join(dest_dir, image)
        shutil.move(src_path, dest_path)

    print(f"Moved {len(sampled_images)} images from {source_dir} to {dest_dir}")

# Paths to the BigOne dataset directories
source_dir = "/ssd_scratch/chirag_saigunda/dataset/"
dest_dir = "/ssd_scratch/chirag_saigunda/datasetval/"

# Number of images to move
num_images_to_move = 5000

# Move images
move_images(source_dir, dest_dir, num_images_to_move)
