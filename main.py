from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from PIL import Image
import pytesseract
import io
import httpx
import requests
from bs4 import BeautifulSoup
import onnxruntime as ort
import numpy as np
from PIL import Image
from io import BytesIO
import torchvision.transforms as T
import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import torchvision.transforms as T
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Allowed HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Configure path to tesseract executable (adjust this)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Sanjiv Kr Sah\Desktop\fastapi-ocr\tesseract.exe"

# Chat history (for prototype)
chat_history: List[dict] = []

# OpenRouter / AI API config
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class ChatInput(BaseModel):
    message: str

# Utility: simple symptom → specialty mapping
SYMPTOM_TO_SPECIALTY = {
    "chest pain": "cardiologist",
    "shortness of breath": "cardiologist",
    "rash": "dermatologist",
    "skin": "dermatologist",
    "cough": "pulmonologist",
    # add more mappings
}

def guess_specialties_from_text(text: str) -> List[str]:
    specialties = set()
    lower = text.lower()
    for symptom, spec in SYMPTOM_TO_SPECIALTY.items():
        if symptom in lower:
            specialties.add(spec)
    return list(specialties) if specialties else []

# Example web-scraping function (prototype) for doctors in a city + specialty
def scrape_doctors(city: str, specialty: str) -> List[dict]:
    # This is just a dummy example using Medindia’s directory page structure
    url = f"https://www.medindia.net/directories/doctors/index.htm?city={city}&speciality={specialty}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    doctors = []
    # Note: these CSS selectors must be adapted to actual site HTML structure
    for entry in soup.select(".drlist"):  # example selector
        name = entry.select_one(".drlist_name")
        spec = entry.select_one(".drlist_speciality")
        address = entry.select_one(".drlist_address")
        phone = entry.select_one(".drlist_phones")
        doc = {
            "name": name.get_text(strip=True) if name else "",
            "specialty": spec.get_text(strip=True) if spec else specialty,
            "address": address.get_text(strip=True) if address else "",
            "phone": phone.get_text(strip=True) if phone else "",
            "city": city
        }
        doctors.append(doc)
    return doctors

# ROUTES

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    text = pytesseract.image_to_string(image)
    return JSONResponse(content={"extracted_text": text.strip()})

@app.post("/analyze-text")
async def analyze_text(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    extracted_text = pytesseract.image_to_string(image).strip()
    if not extracted_text:
        raise HTTPException(status_code=400, detail="No text found in image.")
    prompt = (
        "Analyze the following medical report text and provide a summary and key insights:\n\n"
        + extracted_text
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error contacting analysis API")
    data = response.json()
    ai_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {"extracted_text": extracted_text, "analysis": ai_text}

@app.post("/health-alerts")
async def health_alerts(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    extracted_text = pytesseract.image_to_string(image).strip()
    if not extracted_text:
        raise HTTPException(status_code=400, detail="No text found in image.")
    prompt = (
        "Based on the following medical report text, generate important health alerts "
        "and preventive health recommendations for the patient:\n\n"
        + extracted_text
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error contacting analysis API")
    data = response.json()
    ai_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {
        "extracted_text": extracted_text,
        "health_alerts_and_recommendations": ai_text
    }

@app.post("/chat")
async def chat_with_bot(chat: ChatInput):
    user_message = chat.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message provided.")
    chat_history.append({"role": "user", "content": user_message})
    payload = {
        "model": "gpt-4o-mini",
        "messages": chat_history,
        "max_tokens": 500,
        "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error contacting chatbot API.")
    data = response.json()
    bot_reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    chat_history.append({"role": "assistant", "content": bot_reply})
    return {"response": bot_reply, "chat_history": chat_history}

@app.post("/suggest-doctors")
async def suggest_doctors(
    file: UploadFile = File(...),
    city: str = Query(..., description="City or location to suggest doctors in")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    extracted_text = pytesseract.image_to_string(image).strip()
    if not extracted_text:
        raise HTTPException(status_code=400, detail="No text found in image.")
    specialties = guess_specialties_from_text(extracted_text)
    if not specialties:
        specialties = ["general physician"]
    suggestions = []
    for spec in specialties:
        docs = scrape_doctors(city, spec)
        suggestions.extend(docs)
    return {
        "extracted_text": extracted_text,
        "specialties_detected": specialties,
        "doctor_suggestions": suggestions[:10]  # limit to top 10
    }
    


# Load the ONNX model
onnx_model_path = r"C:\Users\Sanjiv Kr Sah\Desktop\fastapi-ocr\services\VIT23n_quantmodel.onnx"  # Replace with the actual path to your model
onnx_session = ort.InferenceSession(onnx_model_path)

# Define the necessary preprocessing steps
transform = T.Compose([
    T.Resize((256, 256)),  # Adjust to the input size expected by the model (256x256)
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for ImageNet
])

# Predefined mapping of class indices (starting from 1) to disease names and descriptions
class_to_disease = {
    1: {"name": "Acne", "description": "A common skin condition that causes pimples, blackheads, and cysts, mainly on the face, chest, and back."},
    2: {"name": "Actinic Keratosis", "description": "A precancerous area of thick, scaly, or crusty skin that can lead to skin cancer."},
    3: {"name": "Benign Tumors", "description": "Non-cancerous growths or lumps on the skin that do not spread to other parts of the body."},
    4: {"name": "Bullous", "description": "Skin condition characterized by blisters, typically caused by autoimmune diseases or infections."},
    5: {"name": "Candidiasis", "description": "A fungal infection caused by Candida species, affecting areas like the skin, mouth, and genitals."},
    6: {"name": "Drug Eruption", "description": "A skin reaction to medication, which can range from mild rashes to severe reactions."},
    7: {"name": "Eczema", "description": "A group of conditions that cause inflamed, itchy, red, and dry skin, often due to allergies or environmental factors."},
    8: {"name": "Infestations/Bites", "description": "Skin reactions to insect bites or parasitic infestations, like bedbugs or lice."},
    9: {"name": "Lichen", "description": "A type of rash that forms small, flat, purple or reddish bumps on the skin, often caused by immune system dysfunction."},
    10: {"name": "Lupus", "description": "An autoimmune disease that can cause skin rashes, especially in sun-exposed areas, along with other systemic symptoms."},
    11: {"name": "Moles", "description": "Dark, often raised, spots or growths on the skin that can sometimes indicate skin cancer if they change in size, shape, or color."},
    12: {"name": "Psoriasis", "description": "A chronic autoimmune condition that causes the rapid build-up of skin cells, leading to thick, scaly patches."},
    13: {"name": "Rosacea", "description": "A condition that causes redness and visible blood vessels in the face, often mistaken for acne or other skin issues."},
    14: {"name": "Seborrheic Keratoses", "description": "Non-cancerous growths or lesions that can appear on the skin, often resembling warts."},
    15: {"name": "Skin Cancer", "description": "Cancer that forms in the skin cells, commonly caused by prolonged sun exposure or tanning."},
    16: {"name": "Sun/Sunlight Damage", "description": "Skin damage caused by excessive exposure to UV radiation from the sun, leading to wrinkles, age spots, and even skin cancer."},
    17: {"name": "Tinea", "description": "A fungal infection commonly known as ringworm, affecting the skin, nails, and hair."},
    18: {"name": "Rahes", "description": "Rashes are skin conditions characterized by redness, irritation, and bumps or patches, often caused by allergic reactions, infections or irritation"},
    19: {"name": "Vascular Tumors", "description": "Tumors in the blood vessels that can appear as reddish or purple lumps on the skin."},
    20: {"name": "Vasculitis", "description": "Inflammation of blood vessels that can cause skin discoloration and rashes, often related to autoimmune diseases."},
    21: {"name": "Vitiligo", "description": "A condition that causes the skin to lose its pigment, leading to white patches of skin."},
    22: {"name": "Warts", "description": "Small, raised growths caused by viral infections, typically on the hands, feet, and face."},
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receive an image, preprocess it, run it through the ONNX model, and return predictions.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    
    # Read the uploaded image
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes))

    # Apply preprocessing to the image
    input_image = transform(image).unsqueeze(0).numpy()  # Add batch dimension and convert to numpy

    # Ensure the image is in the format the model expects (NCHW format: batch_size, channels, height, width)
    input_image = np.transpose(input_image, (0, 2, 3, 1))  # Convert from (batch, channels, height, width) to (batch, height, width, channels)
    input_image = input_image.astype(np.float32)  # Ensure the input is in float32 type

    # Run inference on the model
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: input_image})

    # Process the result
    # Example: For classification models, if the result contains logits, apply softmax to convert to probabilities
    result = np.squeeze(result[0])  # Remove batch dimension
    predicted_class = np.argmax(result)  # Get the index of the maximum value (predicted class)

    # Retrieve the corresponding disease name and description from the class_to_disease dictionary
    disease_info = class_to_disease.get(predicted_class + 1, {"name": "Unknown", "description": "Unknown condition"})
    
    # Return the result with name and description
    return JSONResponse(content={
        "prediction": disease_info["name"],
        "description": disease_info["description"]
    })
