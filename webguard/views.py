import json
from . import app
from flask import Flask, render_template, request, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import service as serve
import os

app = Flask(__name__)
cors = CORS(app)


@app.route('/')
def hello_world():  # put application's code here
    return 'web guard'


@app.route('/addnew', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        name = (request.form['title'])
        f.save(secure_filename("input." + f.filename.split('.')[-1]))
        filename = str("input." + f.filename.split('.')[-1])
        serve.extract_frames("input.mp4", "faces/"+name)
        os.remove("input.mp4")
        return jsonify({"success": "new face added"})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename("input." + f.filename.split('.')[-1]))
        filename = str("input." + f.filename.split('.')[-1])
        known_encodings, known_names = serve.load_known_faces()
        # Identify faces in a new image
        name = serve.identify_faces(filename, known_encodings, known_names)
        # serve.extract_frames("input.mp4", "faces/"+name)
        os.remove(filename)
        return jsonify({"success": "detection success", "id": name})

@app.route('/severity', methods=['POST'])
def check_severity():
    abFrames = []
    if request.method == 'POST':
        severity = (request.form['severity'])
        detection = serve.get_severity(severity)
        return jsonify({"success": "severity check success", "result":detection})

if __name__ == '__main__':
    app.run()
