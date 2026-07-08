# Object Detection Using TensorFlow

This repository contains a TensorFlow-based object detection workflow built around a Jupyter notebook. It includes dataset splits in COCO format, model-building code, training utilities, and output folders for generated artifacts.

## Project Layout

- `notebooks/Digit_Object_Detection.ipynb` - main notebook for data loading, model definition, training, and evaluation
- `notebooks/dataset/train` - training images and `_annotations.coco.json`
- `notebooks/dataset/valid` - validation images and annotations
- `notebooks/dataset/test` - test images and annotations
- `models/` - saved model checkpoints and exported models
- `outputs/` - plots, metrics, and other generated results

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The project uses TensorFlow 2.18.0 together with NumPy, OpenCV, Matplotlib, scikit-learn, Albumentations, and Jupyter Notebook.

## Run the Notebook

1. Open `notebooks/Digit_Object_Detection.ipynb` in VS Code or Jupyter.
2. Run the setup cells first so paths, constants, and the dataset checks are initialized.
3. Execute the model-building and training cells in order.
4. Use the later evaluation cells to inspect predictions and outputs.

## Model Architecture

The notebook builds a custom CNN detector rather than using a pre-trained transfer-learning backbone.

### Backbone

The feature extractor is a 5-stage CNN named `CNN_Backbone`. Each stage uses the same reusable block:

- `Conv2D`
- `BatchNormalization`
- `ReLU`
- `MaxPooling2D` on the first four stages only

Filter progression in the backbone:

- 32 filters
- 64 filters
- 128 filters
- 256 filters
- 512 filters on the final stage without pooling

With the default input shape of `(224, 224, 3)`, the backbone reduces the feature map to `(14, 14, 512)`.

### Detection Head

The detector head named `DigitDetector` adds a shared prediction block on top of the backbone output:

- `Conv2D(256, 3x3)`
- `BatchNormalization`
- `ReLU`

It then branches into three output heads:

- `boxes`: `Conv2D(4, 1x1)` for bounding box regression
- `objectness`: `Conv2D(1, 1x1, sigmoid)` for object presence
- `classes`: `Conv2D(NUM_CLASSES, 1x1, softmax)` for class probabilities

This produces grid-based predictions with shapes like `(14, 14, 4)`, `(14, 14, 1)`, and `(14, 14, 10)` for the current configuration.

## Dataset Format

The notebook expects each split to contain an `images/` folder and a COCO annotation file named `_annotations.coco.json`.

Expected structure:

```text
notebooks/dataset/
	train/
		images/
		_annotations.coco.json
	valid/
		images/
		_annotations.coco.json
	test/
		images/
		_annotations.coco.json
```

## GPU Notes

This workspace is configured on native Windows with a CPU-only TensorFlow build. If you want NVIDIA GPU acceleration, use WSL2 with a CUDA-enabled TensorFlow stack or a supported Windows DirectML setup.

## Outputs

Training artifacts, plots, and saved models are written to `outputs/` and `models/`.
