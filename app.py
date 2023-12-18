#!/usr/bin/env python
# coding: utf-8

import subprocess
subprocess.run(['pip', 'install', '-Uqq', 'fastai'])
subprocess.run(['pip', 'install', '-Uqq', 'timm'])

from fastai.vision.all import *
import gradio as gr

def finding(x): return x[0].isupper()

learn = load_learner('model.pkl')

#Classifies an x-ray image and returns probabilities for 'No Finding', 'Obstructive Pulmonary Disease' and 'Pneumonia'
categories = ('No Finding', 'Obstructive Pulmonary Disease', 'Pneumonia')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

#Creates a Gradio interface for image classification
examples = ['normal.jpeg', 'obs.jpeg', 'pneu.jpeg']
intf = gr.Interface(fn=classify_image, inputs='image', outputs='label', examples=examples)
intf.launch(inline=False)