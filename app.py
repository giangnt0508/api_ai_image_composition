from flask import Flask, jsonify, request, abort, url_for
from flask_cors import CORS

import os
import base64
from flask import Flask, jsonify, request, abort, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='images', static_url_path='/static')
CORS(app)

# Ensure the 'images' directory exists
if not os.path.exists('images'):
    os.makedirs('images')

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
        filename = secure_filename(f"image_{len(os.listdir('images')) + 1}.png")
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

if __name__ == '__main__':
    app.run(debug=True)