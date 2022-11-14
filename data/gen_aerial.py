"""
Dataset Generator using Dataset for Object Detection in Aerial Images (DOTA) v1.5 images.

Instead of using full images for joint object detection and classification we limit ourselves to properly cropped
objects for image classification.
"""
import math
import os
import random
import sqlite3
import warnings
from zipfile import ZipFile

import numpy as np
from PIL import Image


def create_database(split='train', image_sizes=None):
    """ Reads all annotations and stores objects in SQLite database. """
    # Connect to and create database
    conn = sqlite3.connect('objects.db')
    conn.execute('CREATE TABLE IF NOT EXISTS objects (file TEXT, src TEXT, label TEXT,'
                 'x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER, x3 INTEGER, y3 INTEGER, x4 INTEGER, y4 INTEGER,'
                 'width INTEGER, height INTEGER, im_width INTEGER, im_height INTEGER, im_channels INTEGER)')
    conn.commit()
    # Read annotations from zipped file
    zip_file = ZipFile(os.path.join('dota', split, 'DOTA-v1.5_{}.zip'.format(split)))
    for path in zip_file.namelist():
        annotations = zip_file.read(path).decode('ascii').splitlines()
        src = annotations[0][12:]  # image source given in first line
        objects = []
        for row in annotations[2:]:
            r = row.rstrip().split(' ')
            file, label = path[:-4], r[8]
            im_width, im_height, im_channels = image_sizes[file]
            geom = [int(r[i][:-2]) for i in range(8)]
            width = max(geom[0::2]) - min(geom[0::2])
            height = max(geom[1::2]) - min(geom[1::2])
            objects.append((file, src, label, *geom, width, height, im_width, im_height, im_channels))
        # Store objects in database
        conn.executemany('INSERT INTO objects VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', objects)
        conn.commit()


def get_image_sizes():
    # Load .zip files with images
    train_paths = [os.path.join('train', p) for p in ['part1.zip', 'part2.zip', 'part3.zip']]
    val_paths = [os.path.join('val', 'part1.zip')]
    zip_files = [ZipFile(os.path.join('dota', p)) for p in train_paths + val_paths]
    # Go over each image in each zipped file
    image_sizes = {}
    for zip_file in zip_files:
        for path in zip_file.namelist():
            if not path.endswith('/'):
                # Extract image size for later reference
                file = zip_file.open(path)
                image = Image.open(file)
                image_sizes[path[7:-4]] = (*image.size, 3 if image.mode == 'RGB' else 1)
    return image_sizes


def find_image(name, zip_files):
    """ Looks for image file in the provided zip archives and extracts it if found. """
    path = 'images/{}.png'.format(name)
    # Look for the file in the zip archives
    file = None
    for zip_file in zip_files:
        if path in zip_file.namelist():
            file = zip_file.open(path)
    if not file:
        warnings.warn('Image {}.png not found.'.format(name))
        return None
    return Image.open(file)


def extract_object_from_image(obj, scale, zip_files):
    """ Extracts traffic sign from full image at requested scale. """
    # Load image and object properties
    image = find_image(obj[0], zip_files)
    width, height = image.size  # Image properties
    x1, y1, x2, y2, x3, y3, x4, y4, w, h = obj[3:13]  # Object properties
    xmin, xmax, ymin, ymax = min(x1, x2, x3, x4), max(x1, x2, x3, x4), min(y1, y2, y3, y4), max(y1, y2, y3, y4)
    # Transform object to correct size
    if w < scale and h < scale:
        warnings.warn('Extracted object width {} and height {} is smaller than target {}.'.format(w, h, scale))
    scale_factor = scale / max(w, h)
    image = image.resize((round(width * scale_factor), round(height * scale_factor)), Image.ANTIALIAS)
    width, height = image.size
    # Find a random position for the 64x64 image
    xstart, xend = round(xmin * scale_factor), round(xmax * scale_factor)
    x = random_pos(xstart, xend, width)
    ystart, yend = round(ymin * scale_factor), round(ymax * scale_factor)
    y = random_pos(ystart, yend, height)
    # Crop the 64x64 object from the complete image
    obj = image.crop((x, y, x + 64, y + 64))
    translation = (xstart - x, ystart - y)
    return np.array(obj), translation


def random_pos(minimum, maximum, size):
    """ Finds a random position that is within the bounds given by object min and max, plus image size. """
    lower = max(0, maximum - 64)
    upper = min(size - 64, minimum)
    return random.randint(lower, upper)


def generate():
    # Set seed for reproducibility
    np.random.seed(42)
    # Create database with object bounding boxes
    image_sizes = get_image_sizes()
    if not os.path.exists('objects.db'):
        create_database(split='train', image_sizes=image_sizes)
        create_database(split='val', image_sizes=image_sizes)
    # Find suitable object classes (at least 75 with no rotation)
    conn = sqlite3.connect('objects.db')
    conn.create_function('ANGLE', 2, lambda y, x: abs(math.degrees(math.atan2(y, x))))
    classes = conn.execute('SELECT label, COUNT(*) AS count FROM objects WHERE ANGLE(y2 - y1, x2 - x1) < 3 '
                           'AND MAX(width, height) >= 64 AND im_width / MAX(width, height) > 4 AND im_height / MAX(width, height) > 4 '
                           'AND MAX(x1,x2,x3,x4) <= im_width AND MAX(y1,y2,y3,y4) <= im_height AND im_channels = 3 '
                           'AND src = "GoogleEarth" GROUP BY label HAVING count >= 75')
    classes = list(classes)
    # Load .zip files with images
    train_paths = [os.path.join('train', p) for p in ['part1.zip', 'part2.zip', 'part3.zip']]
    val_paths = [os.path.join('val', 'part1.zip')]
    zip_files = [ZipFile(os.path.join('dota', p)) for p in train_paths + val_paths]
    # Create empty package contents
    num_instances = 25  # Number of sampled images per class
    images = [[[[] for _ in range(num_instances)] for _ in range(48)] for _ in range(3)]
    labels = [[[[] for _ in range(num_instances)] for _ in range(48)] for _ in range(3)]
    scales = [[[[] for _ in range(num_instances)] for _ in range(48)] for _ in range(3)]
    translations = [[[[] for _ in range(num_instances)] for _ in range(48)] for _ in range(3)]
    # Generate scaled images for each object class
    for index, (label, _) in enumerate(classes):  # 10 different aerial objects
        objects = conn.execute('SELECT * FROM objects WHERE label = ? AND src = "GoogleEarth" AND '
                               'ANGLE(y2 - y1, x2 - x1) < 3 AND im_width / MAX(width, height) > 4 AND im_height / MAX(width, height) > 4 AND '
                               'MAX(x1,x2,x3,x4) <= im_width AND MAX(y1,y2,y3,y4) <= im_height AND im_channels = 3 '
                               'ORDER BY MAX(width, height) DESC LIMIT 75', (label,))
        objects = list(objects)
        # Split aerial objects into training, validation and testing
        splits = objects[0::3], objects[1::3], objects[2::3]
        for i in range(num_instances):  # 25 different images per aerial object
            for j, scale in enumerate(range(64, 16, -1)):  # 48 different scales
                for k in range(3):  # 3 sets (training, validation, testing)
                    obj, translation = extract_object_from_image(splits[k][i], scale, zip_files)
                    images[k][j][i].append(obj)
                    labels[k][j][i].append(index)
                    scales[k][j][i].append(scale)
                    translations[k][j][i].append(translation)
        print('Done with {}'.format(label))
    # Collect metadata, two-dimensional numpy array to avoid pickling
    metadata = np.array([
        ['title', 'Scaled and Translated Image Recognition (STIR) Aerial Object'],
        ['description',
         'Testing data for scale invariance. 9 aerial objects cropped at sizes between 17x17 and 64x64 pixels with random position, constrained by image bounds. Full color recorded by camera.'],
        ['author', 'Thomas R. Altstidl (thomas.r.altstidl@fau.de)'],
        ['license',
         'CC BY NC SA 4.0 modified from Dataset for Object Detection in Aerial Images - https://captain-whu.github.io/DOTA/dataset.html'],
        ['version', '1.0.0'],
        ['date', '31 August 2022']
    ])
    lbldata = np.array([label for (label, _) in classes])
    # Save data file
    imgs = np.array(images).swapaxes(2, 3)
    lbls = np.array(labels).swapaxes(2, 3)
    scls = np.array(scales).swapaxes(2, 3)
    psts = np.array(translations).swapaxes(2, 3)
    np.savez_compressed('aerial.npz', imgs=imgs, lbls=lbls, scls=scls, psts=psts,
                        metadata=metadata, lbldata=lbldata)


if __name__ == '__main__':
    generate()
