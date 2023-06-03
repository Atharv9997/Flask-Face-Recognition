from flask import Flask, request
from flask_cors import CORS
from cv2 import cvtColor, COLOR_BGR2RGB 
from skimage import io
from face_recognition import face_encodings, compare_faces

app = Flask(__name__)
CORS(app)

def readImage(imagePath):
    img = io.imread(imagePath)
    rgb_img = cvtColor(img, COLOR_BGR2RGB)
    return face_encodings(rgb_img)[0]

def recognise(img1, img2):
    img_encoding1 = readImage(img1)
    img_encoding2 = readImage(img2)
    return compare_faces([img_encoding1], img_encoding2)

@app.route("/single", methods=['GET'])
def single_predict():
    data = request.args
    val = recognise(data.get('url1'),data.get('url2'))
    return str(val)

if __name__ == "__main__":
    app.run(debug=False)
