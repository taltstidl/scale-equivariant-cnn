# Traffic Sign Recognition

Code for processing the Mapillary Traffic Sign Dataset and training the corresponding models. The original dataset is available [here](https://www.mapillary.com/dataset/trafficsign) under the CC BY-NC-SA license. We only make use of the fully annotated images and disregard the partially annotated images.

## Data Generation

Wherever you are executing the provided `process.py` script, the following folders and files are expected to be present. Note that only files read by the script are mentioned. Others will exist as provided by the referenced download source.

| Folder or File | Download Source ([here](https://www.mapillary.com/dataset/trafficsign)) |
| -------------- | ------------- |
| `mapillary/`<ul><li>`mtsd_v2_fully_annotated/annotations/*.json`</li><li>`mtsd_v2_fully_annotated/splits/*.txt`</li></ul> | mtsd_fully_annotated_annotation.zip |
| `train.0.zip` | mtsd_fully_annotated_images.train.0.zip
| `train.1.zip` | mtsd_fully_annotated_images.train.1.zip
| `train.2.zip` | mtsd_fully_annotated_images.train.2.zip
| `val.zip` | mtsd_fully_annotated_images.val.zip

The script will likely fail if one of the mentioned files is not present. After checking, execute the following script to generate the `trafficsigns` folder with the images. During that process, two SQLite databases (`train.db` and `val.db`) will also be generated as a by-product.

```bash
python process.py
```

## Model Training
