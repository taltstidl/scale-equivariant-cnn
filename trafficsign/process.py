import argparse
import json
import math
import os
import random
import sqlite3
import warnings
import zipfile

from PIL import Image


def create_database(split='train'):
    """ Reads all json files and stores objects in SQLite database. """
    # Connect to and create database
    conn = sqlite3.connect('{}.db'.format(split))
    conn.execute('CREATE TABLE IF NOT EXISTS objects (file TEXT, key TEXT, label TEXT,'
                 'xmin INTEGER, xmax INTEGER, ymin INTEGER, ymax INTEGER, width INTEGER, height INTEGER)')
    conn.commit()
    # Load file with image list
    file = open(os.path.join('mapillary', 'mtsd_v2_fully_annotated', 'splits', '{}.txt'.format(split)))
    images = file.readlines()
    for image in images:
        image = str.strip(image)
        file = open(os.path.join('mapillary', 'mtsd_v2_fully_annotated', 'annotations', '{}.json'.format(image)))
        annotations = json.load(file)
        for object in annotations['objects']:
            if any(object['properties'].values()):
                continue  # Skip unwanted objects
            xmin = math.floor(object['bbox']['xmin'])
            xmax = math.ceil(object['bbox']['xmax'])
            ymin = math.floor(object['bbox']['ymin'])
            ymax = math.ceil(object['bbox']['ymax'])
            width = xmax - xmin
            height = ymax - ymin
            if width > 64 or height > 64:
                continue  # Skip objects that are "too large"
            geom = [xmin, xmax, ymin, ymax, width, height]
            # Insert object into database
            key, label = object['key'], object['label']
            conn.execute('INSERT INTO objects VALUES (?,?,?,?,?,?,?,?,?)', (image, key, label, *geom))
            conn.commit()


def find_image(name, *zip_files):
    """ Looks for image file in the provided zip archives and extracts it if found. """
    path = 'images/{}.jpg'.format(name)
    # Look for the file in the zip archives
    file = None
    for zip_file in zip_files:
        if path in zip_file.namelist():
            file = zip_file.open(path)
    if not file:
        warnings.warn('Image {}.jpg not found.'.format(name))
        return None
    return Image.open(file)


def find_random_pos(minimum, maximum, size):
    """ Finds a random position that is within the bounds given by object min and max, plus image size. """
    lower = max(0, maximum - 64)
    upper = min(size - 64, minimum)
    return random.randint(lower, upper)


def extract_images(images, path, *zip_files):
    """ Extracts the given images to the provided folder. """
    os.makedirs(path, exist_ok=True)
    # Go over each object contained in images, each a tuple (file, key, label, xmin, xmax, ymin, ymax, width, height)
    for obj in images:
        # Load the full image (can contain multiple traffic signs)
        image = find_image(obj[0], *zip_files)
        width, height = image.size  # Image properties
        xmin, xmax, ymin, ymax = obj[3:7]  # Object properties
        # Compute position of 64x64 cropped object
        x = find_random_pos(xmin, xmax, width)
        y = find_random_pos(ymin, ymax, height)
        # Crop and save the object image
        sign = image.crop((x, y, x + 64, y + 64))
        sign.save(os.path.join(path, '{}.jpg'.format(obj[1])))


def extract_images_for_label(label):
    # Load all zip archives
    train_zips = [zipfile.ZipFile('train.0.zip'), zipfile.ZipFile('train.1.zip'), zipfile.ZipFile('train.2.zip')]
    val_zip = zipfile.ZipFile('val.zip')
    # Load train.db
    conn = sqlite3.connect('train.db')
    images = conn.execute('SELECT * FROM objects WHERE label = ?', (label,)).fetchall()
    conn.close()
    # Split train.db images into training and validation
    random.shuffle(images)
    cutoff = int(round(0.8 * len(images)))  # 80% for training, 20% for validation
    train, val = images[:cutoff], images[cutoff:]
    # Extract training and validation images
    path = os.path.join('signs', 'train', label)
    extract_images(train, path, *train_zips)
    path = os.path.join('signs', 'val', label)
    extract_images(val, path, *train_zips)
    # Load val.db
    conn = sqlite3.connect('val.db')
    images = conn.execute('SELECT * FROM objects WHERE label = ?', (label,)).fetchall()
    conn.close()
    # Extract test images
    path = os.path.join('signs', 'test', label)
    extract_images(images, path, val_zip)


def create_folders():
    """ Reads and filters databases, then writes cropped object images to individual folders. """
    # Create necessary directories
    os.makedirs(os.path.join('signs', 'train'), exist_ok=True)
    os.makedirs(os.path.join('signs', 'val'), exist_ok=True)
    os.makedirs(os.path.join('signs', 'test'), exist_ok=True)
    # Select labels that have at least 100 samples in the training data
    conn = sqlite3.connect('train.db')
    counts = conn.execute('SELECT label, COUNT(*) AS count FROM objects GROUP BY label ORDER BY count DESC').fetchall()
    labels = [c[0] for c in counts if c[1] >= 100][1:]  # All labels with >= 100 samples, except "other-sign"
    conn.close()
    # Extract train, val and test images to folders
    for label in labels:
        extract_images_for_label(label)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Command-line interface for processing Mapillary Traffic Sign Data.')
    parser.add_argument('--create-databases', help='Whether the SQLite annotation databases should be created',
                        type=bool, default=False)
    parser.add_argument('--create-folders', help='Whether the image folders should be created',
                        type=bool, default=True)
    args = parser.parse_args()
    # Create annotation databases if requested (otherwise they need to be present)
    if args.create_databases:
        create_database('train')
        create_database('val')
    # Create image folders if requested (compatible with PyTorch ImageFolder)
    if args.create_folders:
        create_folders()


if __name__ == '__main__':
    main()
