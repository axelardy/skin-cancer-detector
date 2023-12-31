from flask import Flask, render_template, jsonify,request
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
model = YOLO('../skin-cancer-detector/model/skin_cancer_v2.pt')

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    image_np = np.frombuffer(image.read(), np.uint8)
    cv2_img = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
    result = model(cv2_img)
    abb_dict = result[0].names
    top1 = result[0].probs.top1
    probs = result[0].probs.cpu().numpy().data
    percentage = [round(value * 100, 5) for value in probs]
    sum = np.sum(percentage)
    names = list(abb_dict.values())
    result_dict = {'typeOfCancer': abb_dict[top1],'dist' : dict(zip(names,percentage))}

    print('sum:'+str(sum))
    print(result_dict)  
    return jsonify(result_dict)

if __name__ == '__main__':
    app.run(debug=True, port=5000,host='0.0.0.0')


