import os
import base64
import requests
from PIL import Image
from io import BytesIO

# --- Configuration ---
# Default Ollama API address. Ensure Ollama is running on this host/port.
OLLAMA_HOST = "http://localhost:11434"
# The name of the multimodal Ollama model you have pulled (e.g., "llava", "bakllava").
OLLAMA_MODEL = "llama3.2-vision:latest" 

# --- IMPORTANT: Configure the path to a sample SROIE image you want to test ---
# This path should be correct relative to where you run this ollama_raw_text_test.py script.
# Based on your folder structure, if this script is in '01 THESIS/',
# and your image is 'X00016469670.jpg' in '01 THESIS/datasets/SROIE2019/test/img/',
# then the path should be:
SAMPLE_IMAGE_FOR_RAW_TEXT_TEST = "../datasets/SROIE2019/test/img/X00016469670.jpg" 
# --- END CONFIGURATION ---

# --- Helper Functions ---

def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG") # Assuming JPEG for SROIE images
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def call_ollama_for_raw_text(image_base64, model_name=OLLAMA_MODEL):
    """
    Calls the Ollama API for a multimodal model to get all visible text.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    headers = {"Content-Type": "application/json"}
    
    # Simple prompt to ask the model to transcribe all text it sees.
    prompt = "Transcribe all visible text from this image. Do not analyze, summarize, or extract specific fields. Just provide the raw text content."

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False # We want the full response at once
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180) # Increased timeout slightly for safety
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        full_response = response.json()
        
        # The raw text response from Ollama usually comes directly in the "response" field
        return full_response.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API for raw text: {e}")
        print("Please ensure Ollama is running and the model is pulled (`ollama run {model_name}`).")
        return None

# --- Main Test Execution ---
if __name__ == "__main__":
    print(f"--- Testing raw text extraction with Ollama model: {OLLAMA_MODEL} ---")
    
    if not os.path.exists(SAMPLE_IMAGE_FOR_RAW_TEXT_TEST):
        print(f"Error: Sample image not found at {SAMPLE_IMAGE_FOR_RAW_TEXT_TEST}. Please adjust the 'SAMPLE_IMAGE_FOR_RAW_TEXT_TEST' path.")
    else:
        print(f"Loading sample image: {os.path.basename(SAMPLE_IMAGE_FOR_RAW_TEXT_TEST)}")
        image_b64 = encode_image_to_base64(SAMPLE_IMAGE_FOR_RAW_TEXT_TEST)
        
        if image_b64:
            print("Sending request to Ollama...")
            raw_text_output = call_ollama_for_raw_text(image_b64, OLLAMA_MODEL)
            
            if raw_text_output:
                print("\n--- Ollama's Raw Text Output ---")
                print(raw_text_output)
                print("---------------------------------")
                print("\n**Analysis:**")
                print("Compare this output carefully to the actual text visible on the image.")
                print(" - If the text is largely incorrect or garbled, the model's fundamental OCR/visual perception for this type of document is likely poor.")
                print(" - If the text is largely accurate, but the previous KIE evaluation was poor, then the problem lies in the model's higher-level understanding and ability to map text segments to specific structured fields (company, date, address, total) within a document's layout.")
            else:
                print("Failed to get raw text output from Ollama (check Ollama server and model).")
        else:
            print("Failed to encode image to base64.")