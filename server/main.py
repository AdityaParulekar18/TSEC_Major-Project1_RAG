import io
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
import tensorflow as tf
import importlib
import os
import json
import re
from dotenv import load_dotenv


# Load variables from .env
load_dotenv()

# Access variables
mistral_key = os.getenv("MISTRAL_API_KEY")

# -------------------
# CONFIG
# -------------------
FAISS_STORE_PATH = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MISTRAL_MODEL = "mistral-small-latest"  # Or mistral-small-latest

MODEL_PATHS = {
    "Wheat": "models/wheat.h5",
    "Rice": "models/rice.h5",
    "Maize": "models/maize.h5"
}

LABELS = {
    "Wheat": {0: "Smut", 1: "Leaf Blight", 2: "Brown Rust", 3: "Healthy"},
    "Rice": {0: "Bacterial Leaf Blight", 1: "Brown Spot", 2: "Leaf Blast", 3: "Healthy"},
    "Maize": {0: "Blight", 1: "Common Rust", 2: "Gray Leaf Spot", 3: "Healthy"}
}

IMG_SIZE = (224, 224)

# -------------------
# INIT APP
# -------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load embeddings & vectorstore
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vector_store = FAISS.load_local(
    FAISS_STORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# Load Mistral API LLM
llm = init_chat_model(MISTRAL_MODEL, model_provider="mistralai")

# -------------------
# HELPERS
# -------------------
def load_model(crop_name: str):
    """Force load using TF's built-in keras."""
    tf_keras_models = importlib.import_module("tensorflow.keras.models")
    return tf_keras_models.load_model(MODEL_PATHS[crop_name], compile=False)

def predict(model, idx_to_class, image: Image.Image):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return idx_to_class[class_idx], confidence


def clean_json_output(text: str):
    # Remove markdown code block formatting if present
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)  # Remove starting ```json or ```
    text = re.sub(r"\n```$", "", text)           # Remove ending ```
    text = text.strip()
    return text


# -------------------
# API
# -------------------
@app.post("/predict")
async def predict_and_rag(
    crop: str = Form(...),
    file: UploadFile = File(...)
):
    # crop = crop.lower()
    # if crop not in MODEL_PATHS:
    #     return {"error": f"Invalid crop '{crop}'. Choose from: {list(MODEL_PATHS.keys())}"}

    # Load relevant CNN
    model = load_model(crop)

    # Read image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Predict disease
    disease, confidence = predict(model, LABELS[crop], image)

    # Retrieve relevant docs from FAISS
    query = f"Crop: {crop}, Disease: {disease}"
    docs = vector_store.similarity_search(query, k=3)
    context_text = "\n\n".join([d.page_content for d in docs])

    # Create prompt for LLM
    template = """
You are an agricultural expert. Based on the following crop disease prediction:

Crop: {crop}
Disease: {disease}
Confidence: {confidence}

Here is relevant info from the knowledge base:
{context}

Return the answer as a JSON object with exactly these keys:
- description: a farmer-friendly explanation of the disease. 3-4 lines for each key.
- cause: causes of the disease.
- treatment: recommended treatments for the disease.
- prevention: preventive measures to avoid the disease.

Ensure the output is valid JSON with double quotes around keys and values.
"""
    prompt = PromptTemplate(
        input_variables=["crop", "disease", "confidence", "context"],
        template=template
    )

    final_prompt = prompt.format(
        crop=crop.capitalize(),
        disease=disease,
        confidence=round(confidence, 4),
        context=context_text
    )

    # Get response from Mistral API
    response = llm.invoke(final_prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)

    # Clean and parse JSON
    cleaned_text = clean_json_output(raw_text)

    try:
        structured_data = json.loads(cleaned_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM output as JSON", "raw_response": raw_text}

    return {
        "crop": crop,
        "disease": disease,
        "confidence": round(confidence, 4),
        **structured_data
    }

# -------------------
# RUN
# -------------------
if __name__ == "__main__":
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("❌ MISTRAL_API_KEY not set. Please export it before running.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
