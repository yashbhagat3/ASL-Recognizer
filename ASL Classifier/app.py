import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename
import pandas as pd
import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy as cp
import numpy as np
import math
from keras.models import load_model
import tensorflow as tf
UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/static/'
ALLOWED_EXTENSIONS = {'jpg','png'}

app = Flask(__name__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# limit upload size upto 8mb
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            prediction = process_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)
            path=str(filename)
            # print(path)
            return render_template('result.html', output = prediction, path=str(filename))
    return render_template('index.html')

global graph
graph = tf.get_default_graph()
model = load_model('model_edged.h5')


def process_file(path, filename):
    images = []
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blured = cv2.GaussianBlur(gray, (5, 5), 0)
    blured = cv2.erode(blured, None, iterations=2)
    blured = cv2.dilate(blured, None, iterations=2)

    high_thresh, thresh_im = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_thresh2=high_thresh*0.5
    low_thresh = 0.1*high_thresh
    edged=~img

    edged=cv2.Canny(edged,200,20)
    model_image =~edged
    model_image = cv2.resize(model_image,dsize=(100,100),interpolation=cv2.INTER_CUBIC)

    images.append(model_image)


    images=np.asarray(images)
    images = images.astype('float32')/255.0

    X_test=images

    X_test = X_test.reshape(X_test.shape + (1,))

    with graph.as_default():
        predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)
    if(predictions==28):
            labe='space'
    elif(predictions==26):
            labe='del'
    elif(predictions==27):
            labe='nothing'
    else:        
        labe=chr(65+predictions)
    

    return labe

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
