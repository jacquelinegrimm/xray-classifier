#!/usr/bin/env python
# coding: utf-8

import subprocess
subprocess.run(['pip', 'install', '-Uqq', 'fastai'])
subprocess.run(['pip', 'install', '-Uqq', 'timm'])

from fastai.vision.all import *
import gradio as gr

# Define a custom label function
def diagnosis(x): return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('Degenerative Infectious Disease', 'Mediastinal Anomalies', 'No Finding', 'Obstructive Pulmonary Disease', 'Pneumonia')

# Function that predicts an image's category and returns a dictionary mapping categories to their probabilities
def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

# Create and launch a Gradio interface, allowing interactive image classification
examples = ['normal.jpeg', 'obs.jpeg', 'pneu.jpeg']
intf = gr.Interface(fn=classify_image, inputs='image', outputs='label', examples=examples)
intf.launch(inline=False)