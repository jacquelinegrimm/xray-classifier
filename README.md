# Chest X-Ray Classification

This repository contains resources to train a model for classifying chest X-rays using a Vision Transformer.

## Files

- `chest-x-ray-classification-vit.ipynb`: Jupyter Notebook for training the classification model.
- `chest-x-ray-practice-segmentation.ipynb`: A draft Jupyter Notebook for image segmentation. Not integrated into the current model version.
- `app.py`: Code for deploying a demo using Gradio.

## Dataset

The model is based on the `vit_base_patch16_224.orig_in21k` architecture and fine-tuned using a segment of a dataset uploaded to Kaggle by Fernando Feltrin. The dataset can be found [here](https://www.kaggle.com/datasets/fernando2rad/x-ray-lung-diseases-images-9-classes).

### Classes

The model was trained to classify chest X-rays into three classes:
1. Normal
2. Pneumonia
3. Obstructive Pulmonary Disease

## Usage

1. Use `chest-x-ray-classification-vit.ipynb` to train the classification model.
2. `app.py` contains code for deploying a demo using Gradio for easy testing and visualization.
3. Check out the in-progress demo at [HuggingFace Spaces](https://huggingface.co/spaces/jacquelinegrimm/xray-classifier). Please note that this is a demo and should not be used for diagnosis.
