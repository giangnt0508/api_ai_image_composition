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
from rembg import remove

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

        # Load the image
        image = cv2.imread(filepath)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for white and black
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        
        # Create masks for white and black
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_white, mask_black)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Convert the hex color to BGR
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        bgr = (rgb[2], rgb[1], rgb[0])
        
        # Create a color overlay
        overlay = np.full(image.shape, bgr, dtype=np.uint8)
        
        # Blend the original image with the color overlay using the mask
        result = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
        colored_cars = cv2.bitwise_and(overlay, overlay, mask=mask)
        result = cv2.add(result, colored_cars)
        
        # Save the result
        result_filename = f"result_{filename}"
        result_filepath = os.path.join('images', result_filename)
        cv2.imwrite(result_filepath, result)

        # Generate URL for the result image
        result_url = url_for('static', filename=result_filename, _external=True)

        return jsonify({"url": result_url}), 200

@app.route('/merge-picture', methods=['POST'])
def merge_picture():
    if 'person_image' not in request.files or 'landscape_image' not in request.files:
        return jsonify({"error": "Both person and landscape images are required"}), 400

    person_file = request.files['person_image']
    landscape_file = request.files['landscape_image']

    if person_file.filename == '' or landscape_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Process person image (ImageBlob)
        person_filename = secure_filename(person_file.filename)
        person_filepath = os.path.join('images', person_filename)
        person_file.save(person_filepath)
        person_img = cv2.imread(person_filepath)

        # Process landscape image (ImageBlob)
        landscape_filename = secure_filename(landscape_file.filename)
        landscape_filepath = os.path.join('images', landscape_filename)
        landscape_file.save(landscape_filepath)
        landscape_img = cv2.imread(landscape_filepath)

        # Debug: Save intermediate images
        cv2.imwrite(os.path.join('images', 'debug_person.png'), person_img)
        cv2.imwrite(os.path.join('images', 'debug_landscape.png'), landscape_img)

        # Remove background from person image
        person_no_bg = remove(person_img)

        # Debug: Save the person image without background
        cv2.imwrite(os.path.join('images', 'debug_person_no_bg.png'), person_no_bg)

        # Resize person image to fit in landscape
        person_height, person_width = person_no_bg.shape[:2]
        landscape_height, landscape_width = landscape_img.shape[:2]
        scale = min(landscape_height / person_height, landscape_width / person_width) * 0.8
        new_height = int(person_height * scale)
        new_width = int(person_width * scale)
        person_resized = cv2.resize(person_no_bg, (new_width, new_height))

        # Debug: Save the resized person image
        cv2.imwrite(os.path.join('images', 'debug_person_resized.png'), person_resized)

        # Position person in landscape (bottom center)
        y_offset = landscape_height - new_height
        x_offset = (landscape_width - new_width) // 2

        # Create a mask for the person
        person_gray = cv2.cvtColor(person_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(person_gray, 0, 255, cv2.THRESH_BINARY)

        # Debug: Save the mask
        cv2.imwrite(os.path.join('images', 'debug_mask.png'), mask)

        # Create ROI in landscape
        roi = landscape_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width]

        # Blend person into landscape
        person_fg = cv2.bitwise_and(person_resized, person_resized, mask=mask)
        landscape_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
        merged = cv2.add(person_fg, landscape_bg)

        # Put merged result back into landscape image
        landscape_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = merged

        # Save result
        result_filename = f"merged_{landscape_filename}"
        result_filepath = os.path.join('images', result_filename)
        cv2.imwrite(result_filepath, landscape_img)

        # Generate URL for the result image
        result_url = url_for('static', filename=result_filename, _external=True)

        return jsonify({"url": result_url}), 200

    except Exception as e:
        return jsonify({"error": f"Error processing images: {str(e)}"}), 400


if __name__ == '__main__':
    app.run(debug=True)