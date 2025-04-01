from sentence_transformers import SentenceTransformer
import os
from flask import Flask, request, render_template, jsonify
from pdfminer.high_level import extract_text
import docx

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load AI Model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def calculate_similarity(resume_text, job_desc):
    embeddings = model.encode([resume_text, job_desc])
    similarity_score = (embeddings[0] @ embeddings[1]) / (sum(embeddings[0]**2)**0.5 * sum(embeddings[1]**2)**0.5)
    return round(similarity_score * 100, 2)  # Convert to percentage

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_resume():
    if "file" not in request.files or "job_desc" not in request.form:
        return jsonify({"error": "No file or job description provided"}), 400

    file = request.files["file"]
    job_desc = request.form["job_desc"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)








    if file.filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file_path)
    elif file.filename.endswith(".docx"):
        resume_text = extract_text_from_docx(file_path)
    else:
        return jsonify({"error": "Unsupported file type"}), 400


    similarity_score = calculate_similarity(resume_text, job_desc)

    return jsonify({
        "message": f"File '{file.filename}' uploaded successfully!",
        "AI_match_score": f"{similarity_score}%",
        "suggestions": "Improve alignment by adding job-related skills and experience."
    })

if __name__ == "__main__":
    app.run(debug=True)
