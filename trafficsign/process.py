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
    random.seed(42)
    # Load all zip archives
    train_zips = [zipfile.ZipFile('train.0.zip'), zipfile.ZipFile('train.1.zip'), zipfile.ZipFile('train.2.zip')]
    val_zip = zipfile.ZipFile('val.zip')
    # Compose queries
    query_sm = 'SELECT *, MAX(width, height) AS size FROM objects WHERE label = ? AND size >= 9 AND size < 37'
    query_lg = 'SELECT *, MAX(width, height) AS size FROM objects WHERE label = ? AND size >= 37 AND size < 65'
    # Load train.db
    conn = sqlite3.connect('train.db')
    train_sm = conn.execute(query_sm, (label,)).fetchall()
    random.shuffle(train_sm)
    train_lg = conn.execute(query_lg, (label,)).fetchall()
    random.shuffle(train_lg)
    conn.close()
    # Extract training images
    extract_images(train_sm[:40], os.path.join('signs', 'evaluation1', 'train', label), *train_zips)
    extract_images(train_sm[:40], os.path.join('signs', 'evaluation3', 'train', label), *train_zips)
    extract_images(train_lg[:40], os.path.join('signs', 'evaluation1', 'train', label), *train_zips)
    extract_images(train_lg[:40], os.path.join('signs', 'evaluation2', 'train', label), *train_zips)
    # Extract validation images
    extract_images(train_sm[40:50], os.path.join('signs', 'evaluation1', 'val', label), *train_zips)
    extract_images(train_sm[40:50], os.path.join('signs', 'evaluation3', 'val', label), *train_zips)
    extract_images(train_lg[40:50], os.path.join('signs', 'evaluation1', 'val', label), *train_zips)
    extract_images(train_lg[40:50], os.path.join('signs', 'evaluation2', 'val', label), *train_zips)
    # Load val.db
    conn = sqlite3.connect('val.db')
    val_sm = conn.execute(query_sm, (label,)).fetchall()
    val_lg = conn.execute(query_lg, (label,)).fetchall()
    conn.close()
    # Extract test images
    extract_images(val_sm[:10], os.path.join('signs', 'evaluation1', 'test', label), val_zip)
    extract_images(val_sm[:10], os.path.join('signs', 'evaluation2', 'test', label), val_zip)
    extract_images(val_lg[:10], os.path.join('signs', 'evaluation1', 'test', label), val_zip)
    extract_images(val_lg[:10], os.path.join('signs', 'evaluation3', 'test', label), val_zip)


def to_dict(counts):
    """ Converts list of tuples (label, size, count) to dict structure dict[label][size] = count. """
    dictionary = {}
    for count in counts:
        label, size, count = count
        if label not in dictionary:
            dictionary[label] = {}
        dictionary[label][size] = count
    return dictionary


def has_enough_samples(label, train_counts, val_counts):
    """ Checks whether enough samples are present. """
    if label not in train_counts or label not in val_counts:
        return False  # Label is not present in both training and validation
    train_counts, val_counts = train_counts[label], val_counts[label]
    # For training/validation, at least 50 samples in the smaller and larger halves
    train_count_sm = sum([train_counts.get(s, 0) for s in range(9, 37)])  # Smaller half of scales
    train_count_lg = sum([train_counts.get(s, 0) for s in range(37, 65)])  # Larger half of scales
    # For testing, at least 10 samples in the smaller and larger halves
    val_count_sm = sum([val_counts.get(s, 0) for s in range(9, 37)])  # Smaller half of scales
    val_count_lg = sum([val_counts.get(s, 0) for s in range(37, 65)])  # Larger half of scales
    return train_count_sm >= 50 and train_count_lg >= 50 and val_count_sm >= 10 and val_count_lg >= 10


def select_labels():
    """ Selects labels that have suitable amounts of samples available. """
    query = 'SELECT label, MAX(width, height) AS size, COUNT(*) AS count FROM objects GROUP BY label, size'
    # Query number of samples per label and size for training images
    conn = sqlite3.connect('train.db')
    train_counts = to_dict(conn.execute(query).fetchall())
    conn.close()
    # Query number of samples per label and size for validation images
    conn = sqlite3.connect('val.db')
    val_counts = to_dict(conn.execute(query).fetchall())
    conn.close()
    # Go over each possible label
    labels = set(list(train_counts.keys()) + list(val_counts.keys()))
    labels = [l for l in labels if has_enough_samples(l, train_counts, val_counts)]
    return labels


def create_folders():
    """ Writes cropped object images to individual folders. """
    labels = select_labels()
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
