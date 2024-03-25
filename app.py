from model.model import Net
from model.predict import get_image_prediction
import torch
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
allowed_file_extension = ('jpg', 'jpeg')

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_file_extension

def upload_file(file):
    filename = secure_filename(file.filename)
    extension = filename.rsplit(".", 1)[1].lower()
    path = "./uploaded_images/img_upload." + extension
    file.save(path)
    return path


@app.route('/', methods=['GET', 'POST'])
def handle_request():
    if request.method == "GET":
        return render_template("html/index.html")
    elif request.method == "POST":
        file = request.files['imagefile']
        if file:
            if allowed_file(file.filename):
                path = upload_file(file)
                model = Net()
                model.load_state_dict(torch.load("./model/model_state_dict.pth"))
                c, confidence = get_image_prediction(path)
                confidence = f'{confidence:.2%}'
                os.remove(path) #delete image after prediction
                return render_template("html/index.html", classification=c, confidence=confidence)
            else:
                return render_template("html/index.html", warn=True)
        return render_template("html/index.html")
    
if __name__ == "__main__":
    app.run(port=5000, debug=True)