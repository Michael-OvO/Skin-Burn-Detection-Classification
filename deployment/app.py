import io
from operator import truediv
import os
import json
from PIL import Image

import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# finds the model inside your directory automatically - works only if there is one model
def find_model():
    for f  in os.listdir():
        if f.endswith(".pt"):
            return f
    print("please place a model file in this directory!")
    
model_name = find_model()
# Load model from the local repository to avoid network dependency
model = torch.hub.load('.', 'custom', model_name, source='local')

model.eval()

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images
# Inference
    results = model(imgs, size=640)  # includes NMS
    return results

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
            
        img_bytes = file.read()
        results = get_prediction(img_bytes)
        results.save(save_dir='static')
        filename = 'image0.jpg'
        
        return render_template('result.html',result_image = filename,model_name = model_name)

    return render_template('index.html',model_name=model_name)
@app.route('/detect', methods=['GET', 'POST'])
def handle_video():
    # some code to be implemented later
    pass

@app.route('/webcam', methods=['GET', 'POST'])
def web_cam():
    # some code to be implemented later
    pass

