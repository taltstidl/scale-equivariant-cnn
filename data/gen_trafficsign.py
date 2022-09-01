"""
Dataset Generator using Mapillary Traffic Sign Dataset v2 images.

Instead of using full images for joint object detection and classification we limit ourselves to properly cropped
traffic signs for image classification.
"""
import json
import math
import os
import random
import sqlite3
import warnings
from zipfile import ZipFile

import numpy as np
from PIL import Image


def create_database(split='train'):
    """ Reads all json files and stores objects in SQLite database. """
    # Connect to and create database
    conn = sqlite3.connect('signs.db')
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
            geom = [xmin, xmax, ymin, ymax, width, height]
            # Insert object into database
            key, label = object['key'], object['label']
            conn.execute('INSERT INTO objects VALUES (?,?,?,?,?,?,?,?,?)', (image, key, label, *geom))
            conn.commit()


def find_image(name, zip_files):
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


def extract_sign_from_image(sign, scale, zip_files):
    """ Extracts traffic sign from full image at requested scale. """
    # Load image and object properties
    image = find_image(sign[0], zip_files)
    width, height = image.size  # Image properties
    xmin, xmax, ymin, ymax, w, h = sign[3:9]  # Object properties
    # Transform object to correct size
    if w < scale:
        warnings.warn('Extracted sign width {} is smaller than target {}.'.format(w, scale))
    if h < scale:
        warnings.warn('Extracted sign height {} is smaller than target {}.'.format(h, scale))
    image = image.resize((round(width * scale / w), round(height * scale / h)), Image.ANTIALIAS)
    width, height = image.size
    # Find a random position for the 64x64 image
    xstart = round(xmin * scale / w)
    x = random_pos(xstart, xstart + scale, width)
    ystart = round(ymin * scale / h)
    y = random_pos(ystart, ystart + scale, height)
    # Crop the 64x64 sign object from the complete image
    sign = image.crop((x, y, x + 64, y + 64))
    translation = (xstart - x, ystart - y)
    return np.array(sign), translation


def random_pos(minimum, maximum, size):
    """ Finds a random position that is within the bounds given by object min and max, plus image size. """
    lower = max(0, maximum - 64)
    upper = min(size - 64, minimum)
    return random.randint(lower, upper)


def generate():
    # Set seed for reproducibility
    np.random.seed(42)
    # Create database with sign bounding boxes
    if not os.path.exists('signs.db'):
        create_database(split='train')
        create_database(split='val')
    # Find suitable object classes (at least 75 with width equals height)
    conn = sqlite3.connect('signs.db')
    classes = conn.execute('SELECT label, COUNT(*) AS count FROM objects WHERE ABS(width - height) < 5 '
                           'GROUP BY label HAVING count >= 75')
    classes = list(classes)
    classes.remove([c for c in classes if c[0] == 'other-sign'][0])  # other-sign is just "catch all" class
    collected_signs = []  # list of actual classes used after filtering
    # Load .zip files with images
    paths = ['train.0.zip', 'train.1.zip', 'train.2.zip', 'val.zip']
    zip_files = [ZipFile(os.path.join('mapillary', p)) for p in paths]
    # Create empty package contents
    num_instances = 25  # Number of sampled images per class
    images = [[[[] for _ in range(num_instances)] for _ in range(48)] for _ in range(3)]
    labels = [[[[] for _ in range(num_instances)] for _ in range(48)] for _ in range(3)]
    scales = [[[[] for _ in range(num_instances)] for _ in range(48)] for _ in range(3)]
    translations = [[[[] for _ in range(num_instances)] for _ in range(48)] for _ in range(3)]
    # Generate scaled images for each object class
    current_sign = -1
    for index, (label, _) in enumerate(classes):  # 16 different traffic signs
        objects = conn.execute('SELECT * FROM objects WHERE label = ? AND ABS(width - height) < 5 '
                               'ORDER BY MIN(width, height) DESC LIMIT 75', (label,))
        objects = list(objects)
        # Check whether it's possible to only downscale traffic signs
        got_sizes = np.array([min(o[7], o[8]) for o in objects])
        if np.any(got_sizes < 64):
            continue
        # Increment current sign after filtering
        collected_signs.append(label)
        current_sign += 1
        # Split traffic signs into training, validation and testing
        splits = objects[0::3], objects[1::3], objects[2::3]
        for i in range(num_instances):  # 25 different images per traffic sign
            for j, scale in enumerate(range(64, 16, -1)):  # 48 different scales
                for k in range(3):  # 3 sets (training, validation, testing)
                    sign, translation = extract_sign_from_image(splits[k][i], scale, zip_files)
                    images[k][j][i].append(sign)
                    labels[k][j][i].append(current_sign)
                    scales[k][j][i].append(scale)
                    translations[k][j][i].append(translation)
    # Collect metadata, two-dimensional numpy array to avoid pickling
    metadata = np.array([
        ['title', 'Scaled and Translated Image Recognition (STIR) Traffic Sign'],
        ['description',
         'Testing data for scale invariance. 20 traffic signs cropped at sizes between 17x17 and 64x64 pixels with random position, constrained by image bounds. Full color recorded by camera.'],
        ['author', 'Thomas R. Altstidl (thomas.r.altstidl@fau.de)'],
        ['license',
         'CC BY NC SA 4.0 modified from Mapillary Traffic Sign Dataset - https://www.mapillary.com/dataset/trafficsign'],
        ['version', '1.0.0'],
        ['date', '24 May 2022']
    ])
    lbldata = np.array(collected_signs)
    # Save data file
    imgs = np.array(images).swapaxes(2, 3)
    lbls = np.array(labels).swapaxes(2, 3)
    scls = np.array(scales).swapaxes(2, 3)
    psts = np.array(translations).swapaxes(2, 3)
    np.savez_compressed('trafficsign.npz', imgs=imgs, lbls=lbls, scls=scls, psts=psts,
                        metadata=metadata, lbldata=lbldata)


if __name__ == '__main__':
    generate()
