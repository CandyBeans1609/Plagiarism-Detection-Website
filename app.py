from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Define path to current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models using relative paths
try:
    with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    tfidf_vectorizer = None

def detect(input_text):
    if not model or not tfidf_vectorizer:
        return "Model not loaded"
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism Detected"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form.get('text', '')
    detection_result = detect(input_text)
    return render_template('index.html', result=detection_result)

if __name__ == "__main__":
    app.run(debug=True)
