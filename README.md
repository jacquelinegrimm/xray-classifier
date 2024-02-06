# Chest X-Ray Classification

This repository contains resources to train a model for classifying chest X-rays using a Vision Transformer.

## Files

- `chest-x-ray-classification-vit.ipynb`: Jupyter Notebook for training the classification model.
- `app.py`: Code for deploying a demo using Gradio.

## Dataset

The model is based on the `vit_base_patch16_224.orig_in21k` architecture and fine-tuned using a segment of a dataset uploaded to Kaggle by Fernando Feltrin. The dataset can be found [here](https://www.kaggle.com/datasets/fernando2rad/x-ray-lung-diseases-images-9-classes).

### Classes

The model was trained to classify chest X-rays into five classes:
1. Normal
2. Pneumonia
3. Obstructive Pulmonary Disease (emphysema, bronchopneumonia, bronchiectasis, embolism)
4. Degenerative Infectious Disease (tuberculosis, sarcoidosis, proteinosis, fibrosis)
5. Mediastinal Anomalies (pericarditis, arteriovenous malformations, lymph node enlargement)

## Usage

1. Use `chest-x-ray-classification-vit.ipynb` to train the classification model.
2. `app.py` contains code for deploying a demo using Gradio for easy testing and visualization.
3. Check out the in-progress demo at [HuggingFace Spaces](https://huggingface.co/spaces/jacquelinegrimm/xray-classifier).
