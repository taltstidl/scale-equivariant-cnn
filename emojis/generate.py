"""
Dataset Generator using Font Awesome icons.

To build your own list of the icons currently displayed on the website execute the following in the JavaScript console:
>> console.log('[\'' + Array.from(document.querySelectorAll('.select-all')).map(x => x.innerText).join('\', \'') + '\']')
"""
import argparse
import io
import os
import sys

import cairosvg
import numpy as np
from PIL import Image, ImageOps

# Predefined list of emoji icons
emojis = ['tired', 'surprise', 'smile-wink', 'smile-beam', 'sad-tear', 'sad-cry', 'meh-rolling-eyes', 'meh-blank', 'laugh-wink', 'laugh-squint', 'laugh-beam', 'laugh', 'kiss-wink-heart', 'kiss-beam', 'kiss', 'grin-wink', 'grin-tongue-wink',
          'grin-tongue-squint', 'grin-tongue', 'grin-tears', 'grin-stars', 'grin-squint-tears', 'grin-squint', 'grin-hearts', 'grin-beam-sweat', 'grin-beam', 'grin-alt', 'grin', 'grimace', 'frown-open', 'flushed', 'dizzy', 'angry', 'smile', 'meh', 'frown']


def generate(icons):
    file_name = 'emoji.npz'
    image_size = 64  # 64x64 pixels
    num_scales = 32  # 32 different scales
    # Delete old data file
    if os.path.exists(file_name):
        os.remove(file_name)
    # Set seed for reproducibility
    np.random.seed(42)
    # Determine scales and translations for all 3 sets (training, validation, testing)
    scale = np.tile(np.arange(64, 32, -1), 3)
    translate_x = np.floor(np.random.random_sample(3 * num_scales) * (image_size - scale + 1)).astype(int)
    translate_y = np.floor(np.random.random_sample(3 * num_scales) * (image_size - scale + 1)).astype(int)
    # Pre-generate black and white images (needed for composite)
    black = Image.new('RGBA', size=(image_size, image_size), color='#000')
    white = Image.new('RGBA', size=(image_size, image_size), color='#fff')
    # Generate all icons
    images, labels = [], []
    for j in range(3):  # 3 sets (training, validation, testing)
        for i in range(num_scales):  # 32 different scales
            for index, icon in enumerate(icons):  # 36 different emojis
                path = os.path.join('fontawesome', 'svgs', 'regular', '{}.svg'.format(icon))
                # Render emoji at given size to the buffer
                mem = io.BytesIO()
                cairosvg.svg2png(url=path, write_to=mem, parent_width=scale[i], parent_height=scale[i])
                # Compose full image
                image = Image.open(mem)
                mask = Image.new('RGBA', size=(image_size, image_size), color='#0000')
                mask.paste(image, box=(translate_x[j * num_scales + i], translate_y[j * num_scales + i]))
                image = Image.composite(white, black, mask)
                # Add scaled and translated icon to data
                images.append(np.array(ImageOps.grayscale(image)))
                labels.append(index)
    # Collect latent scales and translations
    scales = np.repeat(scale, len(icons))
    translations = np.repeat(np.stack([translate_x, translate_y], axis=1), len(icons), axis=0)
    # Collect metadata, two-dimensional numpy array to avoid pickling
    metadata = np.array([
        ['title', 'Scaled and Translated Icon Recognition (STIR)'],
        ['description', 'Testing data for scale invariance. The 36 emojis rendered at sizes between 33x33 and 64x64 pixels with random position, constrained by image bounds. White icon on black background.'],
        ['author', 'Thomas R. Altstidl (thomas.r.altstidl@fau.de)'],
        ['license', 'CC BY 4.0 modified from Font Awesome Free 5.15.3 by @fontawesome - https://fontawesome.com (License - https://fontawesome.com/license/free)'],
        ['version', '2.0.0'],
        ['date', '7 April 2021']
    ])
    icondata = np.array(icons)
    # Save data file
    images, labels = np.array(images), np.array(labels)
    np.savez_compressed(file_name, imgs=images, lbls=labels, scls=scales, psts=translations, metadata=metadata, icondata=icondata)


if __name__ == "__main__":
    generate(emojis)
