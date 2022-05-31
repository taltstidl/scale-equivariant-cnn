"""
Dataset Generator using Font Awesome icons.

To build your own list of the icons currently displayed on the website execute the following in the JavaScript console:
>> console.log('[\'' + Array.from(document.querySelectorAll('.select-all')).map(x => x.innerText).join('\', \'') + '\']')
"""
import io
import os
os.add_dll_directory(r"C:\gtk\bin")  # Can be used for Windows

import cairosvg
import numpy as np
from PIL import Image, ImageOps

# Predefined list of emoji icons
emojis = ['tired', 'surprise', 'smile-wink', 'smile-beam', 'sad-tear', 'sad-cry', 'meh-rolling-eyes', 'meh-blank',
          'laugh-wink', 'laugh-squint', 'laugh-beam', 'laugh', 'kiss-wink-heart', 'kiss-beam', 'kiss', 'grin-wink',
          'grin-tongue-wink',
          'grin-tongue-squint', 'grin-tongue', 'grin-tears', 'grin-stars', 'grin-squint-tears', 'grin-squint',
          'grin-hearts', 'grin-beam-sweat', 'grin-beam', 'grin-alt', 'grin', 'grimace', 'frown-open', 'flushed',
          'dizzy', 'angry', 'smile', 'meh', 'frown']


def generate(icons):
    file_name = 'emoji.npz'
    image_size = 64  # 64x64 pixels
    num_scales = 32  # 32 different scales
    # Delete old data file
    if os.path.exists(file_name):
        os.remove(file_name)
    # Set seed for reproducibility
    np.random.seed(42)
    # Pre-generate black and white images (needed for composite)
    black = Image.new('RGBA', size=(image_size, image_size), color='#000')
    white = Image.new('RGBA', size=(image_size, image_size), color='#fff')
    # Create empty package contents
    images = [[[] for _ in range(48)] for _ in range(3)]
    labels = [[[] for _ in range(48)] for _ in range(3)]
    scales = [[[] for _ in range(48)] for _ in range(3)]
    translations = [[[] for _ in range(48)] for _ in range(3)]
    # Generate all icons
    for index, icon in enumerate(icons):  # 36 different emojis
        for i, scale in enumerate(range(64, 16, -1)):  # 32 different scales
            for j in range(3):  # 3 sets (training, validation, testing)
                path = os.path.join('fontawesome', 'svgs', 'regular', '{}.svg'.format(icon))
                # Render emoji at given size to the buffer
                mem = io.BytesIO()
                cairosvg.svg2png(url=path, write_to=mem, parent_width=scale, parent_height=scale)
                # Choose random translation
                translate_x, translate_y = np.random.randint(0, image_size - scale + 1, 2)
                # Compose full image
                image = Image.open(mem)
                mask = Image.new('RGBA', size=(image_size, image_size), color='#0000')
                mask.paste(image, box=(translate_x, translate_y))
                image = Image.composite(white, black, mask)
                # Add scaled and translated icon to data
                images[j][i].append(np.array(ImageOps.grayscale(image)))
                labels[j][i].append(index)
                scales[j][i].append(scale)
                translations[j][i].append((translate_x, translate_y))
    # Collect metadata, two-dimensional numpy array to avoid pickling
    metadata = np.array([
        ['title', 'Scaled and Translated Image Recognition (STIR) Emoji'],
        ['description',
         'Testing data for scale invariance. 36 emojis rendered at sizes between 17x17 and 64x64 pixels with random position, constrained by image bounds. White icon on black background.'],
        ['author', 'Thomas R. Altstidl (thomas.r.altstidl@fau.de)'],
        ['license',
         'CC BY 4.0 modified from Font Awesome Free 5.15.3 by @fontawesome - https://fontawesome.com (License - https://fontawesome.com/license/free)'],
        ['version', '1.0.0'],
        ['date', '24 May 2022']
    ])
    lbldata = np.array(icons)
    # Save data file
    imgs, lbls, scls, psts = np.array(images), np.array(labels), np.array(scales), np.array(translations)
    np.savez_compressed(file_name, imgs=imgs, lbls=lbls, scls=scls, psts=psts,
                        metadata=metadata, lbldata=lbldata)


if __name__ == "__main__":
    generate(emojis)
