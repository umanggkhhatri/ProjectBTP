import os
import logging

# --- MACOS CRASH FIXES (MUST BE AT THE TOP) ---
# These environment variables prevent the "Python quit unexpectedly" error 
# caused by conflicts between Hugging Face tokenizers and macOS memory management.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract

# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- TESSERACT CONFIGURATION ---
# IMPORTANT: If you are on Windows and get a "TesseractNotFoundError", 
# uncomment the line below and update the path to where you installed Tesseract.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- GLOBAL MODEL CACHE ---
# We store loaded models here so we don't reload them on every request.
MODEL_CACHE = {}

def get_llm_generator(model_name="gpt2"):
    """
    Retrieves a text-generation pipeline. Checks cache first to improve speed.
    """
    global MODEL_CACHE
    
    # 1. Check if model is already loaded
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    logger.info(f"Loading model '{model_name}'... This might take a while.")
    
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        
        # Load tokenizer and model
        # invalidating cache to ensure clean load if previous attempts failed
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create pipeline (device=-1 means CPU. Set to 0 if you have a supported GPU)
        gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
        
        # Save to cache
        MODEL_CACHE[model_name] = gen_pipeline
        logger.info(f"Model '{model_name}' loaded successfully.")
        return gen_pipeline
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return None

def ocr_image(path, lang="eng"):
    """
    Extracts text from an image using Tesseract OCR.
    """
    try:
        img = Image.open(path)
        # Using tesseract to convert image to string
        return pytesseract.image_to_string(img, lang=lang).strip()
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        return ""

UNIVERSAL_INSTRUCTION = """You are a helpful, intelligent, and reliable AI assistant.
Your goals:
1. Understand the user's intent clearly.
2. Use the provided OCR text as context if available.
3. Be concise and polite.
"""

def run_assistant(user_text, image_path=None, model_name="gpt2"):
    ocr_text = ""
    
    # 1. Perform OCR if image exists
    if image_path:
        ocr_text = ocr_image(image_path)
        if not ocr_text:
            ocr_text = "No text found in image (or OCR failed)."

    # 2. Build Prompt
    prompt_parts = [UNIVERSAL_INSTRUCTION]
    if ocr_text:
        prompt_parts.append(f"Extracted text from image:\n{ocr_text}")
    prompt_parts.append(f"User message:\n{user_text}")
    
    # Standard prompt format
    prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"

    # 3. Get Model
    gen = get_llm_generator(model_name)
    if gen is None:
        raise RuntimeError(f"Could not load LLM model: {model_name}. check console logs.")

    # 4. Generate Response
    # max_new_tokens: limits response length
    try:
        # Reduced batch size and length slightly to be safer on memory
        out = gen(prompt, max_new_tokens=150, do_sample=True, temperature=0.7, pad_token_id=50256)
        reply = out[0]["generated_text"]
        
        # Clean up: Remove the prompt from the beginning of the reply if present
        if reply.startswith(prompt):
            reply = reply[len(prompt):].strip()
            
    except Exception as e:
        logger.error(f"Generation Error: {e}")
        reply = f"Error during generation: {e}"

    return reply, ocr_text


# ---------- Flask App ----------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
# Limit upload size to 16MB to prevent crashes
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    reply = None
    ocr_text = None
    error = None

    if request.method == "POST":
        user_text = request.form.get("text", "").strip()
        # Use gpt2 if user leaves model field blank
        model_name = request.form.get("model", "gpt2").strip() or "gpt2"
        
        image_file = request.files.get("image")
        image_path = None

        try:
            if image_file and image_file.filename:
                filename = secure_filename(image_file.filename)
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                image_file.save(image_path)

            reply, ocr_text = run_assistant(user_text, image_path, model_name)
            
        except Exception as e:
            error = str(e)
            logger.error(f"App Error: {e}")
        finally:
            # Clean up the uploaded image file after processing
            if image_path and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except Exception as cleanup_error:
                    logger.warning(f"Could not delete temp file: {cleanup_error}")

    return render_template("index.html",
                           reply=reply,
                           ocr_text=ocr_text,
                           error=error)


if __name__ == "__main__":
    # Optional: Pre-load the default model when app starts
    print("Pre-loading default model (gpt2)... Please wait.")
    try:
        get_llm_generator("gpt2")
        print("Ready!")
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")

    app.run(debug=True)