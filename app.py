from flask import Flask, render_template, request
import os
import PyPDF2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords once
nltk.download('stopwords')

app = Flask(__name__)

# ===== ABSOLUTE PATH SETUP (IMPORTANT) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'resumes')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ===== PDF READ FUNCTION =====
def read_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# ===== MAIN ROUTE =====
@app.route('/', methods=['GET', 'POST'])
def index():
    scores = []

    if request.method == 'POST':
        job_description = request.form.get('job_description')
        resumes = request.files.getlist('resumes')

        texts = [job_description]
        resume_names = []

        for resume in resumes:
            if resume.filename == "":
                continue

            save_path = os.path.join(app.config['UPLOAD_FOLDER'], resume.filename)
            resume.save(save_path)

            resume_text = read_pdf(save_path)
            texts.append(resume_text)
            resume_names.append(resume.filename)

        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform(texts)

        similarity_scores = cosine_similarity(vectors[0:1], vectors[1:])[0]

        for i in range(len(similarity_scores)):
            scores.append(
                (resume_names[i], round(similarity_scores[i] * 100, 2))
            )

        scores.sort(key=lambda x: x[1], reverse=True)

    return render_template('index.html', scores=scores)


# ===== RUN APP =====
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
