from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re

app = Flask(__name__)

# Basic skill list (you can expand this)
SKILLS_DB = [
    "python", "java", "c++", "machine learning", "deep learning",
    "tensorflow", "pytorch", "flask", "django",
    "sql", "mongodb", "nlp", "data analysis",
    "pandas", "numpy", "scikit-learn",
    "html", "css", "javascript", "react"
]

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.lower()

def calculate_similarity(resume, job_desc):
    documents = [resume, job_desc]
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(similarity[0][0] * 100, 2), tfidf.get_feature_names_out()

def extract_skills(text):
    found_skills = []
    for skill in SKILLS_DB:
        if skill in text:
            found_skills.append(skill)
    return found_skills

@app.route('/', methods=['GET', 'POST'])
def index():
    score = None
    resume_skills = []
    missing_skills = []
    top_keywords = []

    if request.method == 'POST':

        job_desc = request.form['job_desc'].lower()

        # Handle PDF Upload
        if 'resume_file' in request.files and request.files['resume_file'].filename != '':
            file = request.files['resume_file']
            resume = extract_text_from_pdf(file)
        else:
            resume = request.form['resume'].lower()

        score, keywords = calculate_similarity(resume, job_desc)

        resume_skills = extract_skills(resume)
        job_skills = extract_skills(job_desc)

        missing_skills = list(set(job_skills) - set(resume_skills))
        top_keywords = keywords[:10]

    return render_template(
        'index.html',
        score=score,
        resume_skills=resume_skills,
        missing_skills=missing_skills,
        top_keywords=top_keywords
    )

if __name__ == '__main__':
    app.run(debug=True)