from flask import Flask, render_template, request
import fitz  # PyMuPDF for PDF text extraction
import os
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("resume_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Map job roles to background image filenames
role_images = {
    "Data Science": "data_science.jpg",
    "Software Engineer": "software_engineer.jpg",
    "HR":"HR.jpeg",
    "Testing":"testing.jpg",
    "Civil Engineer":"civil-engineer.jpg",
    "Operations Manager":"Operations-Manager.jpg",
    "Database":"database.jpg",
    "Mechanical Engineer":"mechanical_engineer.jpeg",
    "Network Security Engineer":"Network_security.jpg",
    "Web Developer": "web_developer.jpg",
    "Android Developer": "android_developer.jpg",
    "UI/UX Designer": "ui_ux_designer.jpg",
    "Project Manager": "project_manager.jpg",
    "Business Analyst": "business_analyst.jpg"
}

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return "No file uploaded", 400

    file = request.files['resume']

    if file.filename == '':
        return "No selected file", 400

    if not file.filename.lower().endswith('.pdf'):
        return "Invalid file format. Please upload a PDF.", 400

    # Save the uploaded file
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    # Extract text from the PDF using PyMuPDF
    text = extract_text_from_pdf(filepath)

    # Preprocess and predict
    input_vector = vectorizer.transform([text])
    prediction = model.predict(input_vector)[0]

    # Get image for the predicted job role
    image_file = role_images.get(prediction, "default.jpg")

    # Remove uploaded file after use
    os.remove(filepath)

    # Render result
    return render_template("result.html", prediction=prediction, raw=text, image_file=image_file)

# PDF text extraction function
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Run the app
if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
