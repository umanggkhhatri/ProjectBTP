## ProjectBTP: Local LLM & OCR Assistant

### Overview

ProjectBTP is a **Flask-based web application** that combines **local Large Language Models (LLMs)** with **Optical Character Recognition (OCR)**.  
Users can upload images to extract text and then interact with an AI assistant (default model: `gpt2`) directly from the browser.

### Key Features

- **Local LLM inference**: Runs Hugging Face Transformer models locally (default: `gpt2`).
- **OCR integration**: Extracts text from uploaded images using Tesseract OCR.
- **Context-aware responses**: Uses both your text prompt and OCR-extracted text to generate responses.
- **Model caching**: Loads models once and reuses them for faster subsequent requests.
- **User-friendly web UI**: Simple interface for uploading images and chatting with the model.

### Tech Stack

- **Backend**: Python, Flask
- **ML / NLP**: Hugging Face Transformers (e.g. `gpt2`)
- **OCR**: Tesseract
- **Frontend**: HTML, CSS, JavaScript

---

### Project Structure

At a high level:

- `app.py` — Flask application entry point and API routes.
- `templates/` — HTML templates (e.g. `index.html` for the main UI).
- `static/` — Static assets (`style.css`, `script.js`, images, etc.).
- `uploads/` — Folder where uploaded images/files are stored (ignored by git).

---

### Prerequisites

Before running the application, make sure you have:

1. **Python 3.9+**
2. **Tesseract OCR Engine** installed on your system:
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from the official Tesseract site and add it to your PATH.

---

### How to Run the Project Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/ProjectBTP.git
   cd ProjectBTP
   ```

2. **(Optional but recommended) Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   If you have a `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Or, minimally:

   ```bash
   pip install flask transformers torch pillow pytesseract
   ```

4. **Set any required environment variables (if applicable)**

   For example (customize to your needs):

   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development  # optional
   ```

5. **Run the Flask app**

   ```bash
   python app.py
   ```

   By default, the app should be available at:  
   `http://127.0.0.1:5000/`

---

### Usage

1. Open the web app in your browser.
2. Upload an image containing text (for OCR).
3. Optionally enter additional text / questions in the prompt box.
4. Submit and read the model’s response, which uses both the OCR text and your prompt.

---

### Contributors

- **Umang Khatri** — 2403AI03  
- **Shivang Karthikey** — 2403AI02  
- **Gagan Yadav** — 2403AI05  

---
