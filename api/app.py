from flask import Flask, request, render_template_string, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K
from PIL import Image
import io
import os

app = Flask(__name__)

# --- Custom F1Score Metric ---
class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# --- Load trained model ---
model_path = r'MobileNetV2_best_model.h5'
model = tf.keras.models.load_model(model_path, custom_objects={'f1_score': F1Score})

# --- Serve favicon ---
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

# --- Image preprocessing ---
def preprocess_image(image_file, target_size=(150, 150)):
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# --- Tailwind HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fake vs Real Classifier</title>
    <link rel="icon" href="/favicon.ico" type="image/x-icon">
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-2xl shadow-xl max-w-md w-full">
        <h1 class="text-2xl font-bold mb-6 text-center text-gray-800">🧠 Image Authenticity Classifier</h1>
        <form method="POST" action="/predict" enctype="multipart/form-data" class="space-y-4">
            <input type="file" name="image" accept="image/*" required
                   class="block w-full text-sm text-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"/>
            <button type="submit"
                    class="w-full bg-green-600 text-white py-2 rounded-xl hover:bg-green-700 transition">
                Classify Image
            </button>
        </form>

        {% if result %}
        <div class="mt-6 text-center">
            <p class="text-xl font-semibold text-gray-700">
                Result: <span class="text-indigo-600">{{ result.label|capitalize }}</span>
            </p>
            <p class="text-sm text-gray-500">Confidence: {{ result.confidence }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "Empty filename", 400

    try:
        img_array = preprocess_image(file)
        prediction = model.predict(img_array)
        label = 'fake' if prediction[0] < 0.5 else 'real'
        confidence = round(float(1 - prediction[0] if label == 'fake' else prediction[0]), 2)

        return render_template_string(HTML_TEMPLATE, result={'label': label, 'confidence': confidence})

    except Exception as e:
        return f"Error processing image: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# app = app  # Uncomment this if you're deploying to Vercel
