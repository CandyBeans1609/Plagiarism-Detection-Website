from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
X_train = pickle.load(open('X_train.pkl', 'rb'))

def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)

    if result[0] == 1:
        cosine_sim = cosine_similarity(vectorized_text, X_train).max()
        return (f"Plagiarism Detected<br>"
                f"The text is plagiarized with a similarity score of {cosine_sim * 100:.2f}%")
    else:
        return "No Plagiarism Detected"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']
    detection_result = detect(input_text)
    return render_template('index.html', result=detection_result)

if __name__ == "__main__":
    app.run(debug=True)
