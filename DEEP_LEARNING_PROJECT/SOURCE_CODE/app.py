from __future__ import division,print_function
import sys
import os
import glob
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras import backend
from tensorflow.keras import backend
from tensorflow import keras

import tensorflow as tf

from skimage.transform import resize

from flask import Flask,redirect,url_for,request,render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

model = load_model('Garbage.h5')

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/Image',methods=['POST','GET'])
def prediction():
    return render_template('base.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        predictions_dir = os.path.join(basepath, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)  # Create predictions folder if it doesn't exist
        file_path = os.path.join(predictions_dir, 'images.jpg')
        f.save(file_path)
        img = image.load_img(file_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        predicted_class = np.argmax(preds[0])  # Get the index of the highest probability
        index = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        text = "The Predicted Garbage is " + str(index[predicted_class])

        return text

if __name__ == '__main__':
    app.run(debug=True,threaded=False)
