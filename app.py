from flask import Flask, jsonify, request, abort, url_for
from flask_cors import CORS

import os
import base64
import time
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, abort, url_for
from werkzeug.utils import secure_filename
from apscheduler.schedulers.background import BackgroundScheduler
import cv2
import numpy as np

app = Flask(__name__, static_folder='images', static_url_path='/static')
CORS(app)

# Ensure the 'images' directory exists
if not os.path.exists('images'):
    os.makedirs('images')

def delete_old_images():
    now = datetime.now()
    for filename in os.listdir('images'):
        filepath = os.path.join('images', filename)
        file_creation_time = datetime.fromtimestamp(os.path.getctime(filepath))
        if now - file_creation_time > timedelta(hours=5):
            os.remove(filepath)

# Schedule the delete_old_images function to run every hour
scheduler = BackgroundScheduler()
scheduler.add_job(func=delete_old_images, trigger="interval", hours=1)
scheduler.start()

# Route to save images
@app.route('/save-image', methods=['POST'])
def save_image():
    if 'base64' not in request.json:
        abort(400, description="Missing base64 data")
    
    try:
        # Remove the "data:image/png;base64," prefix if present
        base64_data = request.json['base64'].split(',')[-1]
        
        # Decode the base64 string
        image_data = base64.b64decode(base64_data)
        
        # Generate a unique filename
        filename = secure_filename(f"image_{int(time.time())}.png")
        filepath = os.path.join('images', filename)
        
        # Save the image
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        # Generate the URL for the saved image
        image_url = url_for('static', filename=filename, _external=True)
        
        return jsonify({"url": image_url}), 200
    except Exception as e:
        abort(400, description=f"Error processing image: {str(e)}")

@app.after_request
def add_header(response):
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

@app.route('/change-color', methods=['POST'])
def change_color():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if 'color' not in request.form:
        return jsonify({"error": "No color provided"}), 400

    color = request.form['color']
    if not color.startswith('#') or len(color) != 7:
        return jsonify({"error": "Invalid color format. Use #RRGGBB"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('images', filename)
        file.save(filepath)

        # Load the image and convert to HSV colourspace
        image = cv2.imread(filepath)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define lower and upper limits of what we call "orange"
        orange_lo = np.array([10, 100, 20])
        orange_hi = np.array([30, 255, 255])

        # Mask image to only select oranges
        mask = cv2.inRange(hsv, orange_lo, orange_hi)

        # Convert the hex color to BGR
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        bgr = (rgb[2], rgb[1], rgb[0])

        # Change image to the specified color where we found orange
        image[mask > 0] = bgr

        # Save the result
        result_filename = f"result_{filename}"
        result_filepath = os.path.join('images', result_filename)
        cv2.imwrite(result_filepath, image)

        # Generate URL for the result image
        result_url = url_for('static', filename=result_filename, _external=True)

        return jsonify({"url": result_url}), 200

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

@app.route('/change-car-color', methods=['POST'])
def change_car_color():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if 'color' not in request.form:
        return jsonify({"error": "No color provided"}), 400

    color = request.form['color']
    if not color.startswith('#') or len(color) != 7:
        return jsonify({"error": "Invalid color format. Use #RRGGBB"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('images', filename)
        file.save(filepath)

        # Load the image
        image = cv2.imread(filepath)
        height, width, channels = image.shape

        # Detect objects
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Convert the hex color to BGR
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        bgr = (rgb[2], rgb[1], rgb[0])

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == "car":
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Create a mask for the car
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

                    # Change car color
                    image[mask > 0] = bgr

        # Save the result
        result_filename = f"result_{filename}"
        result_filepath = os.path.join('images', result_filename)
        cv2.imwrite(result_filepath, image)

        # Generate URL for the result image
        result_url = url_for('static', filename=result_filename, _external=True)

        return jsonify({"url": result_url}), 200

if __name__ == '__main__':
    app.run(debug=True)