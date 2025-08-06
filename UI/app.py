from flask import Flask, render_template, request
import os
from utils.inference import predict_and_return_images

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files.get("nifti_file")
        if file and file.filename.endswith(".nii"):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            result_images = predict_and_return_images(filepath)

            return render_template("result.html",
                                   input_img=result_images['input'],
                                   mask_img=result_images['mask'],
                                   overlay_img=result_images['overlay'],
                                   filename=file.filename)
        else:
            return "Invalid file format. Please upload a .nii file.", 400
    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)
