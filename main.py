import requests
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
genai.configure(api_key= "AIzaSyDVkQaXWifIjb9phPmz7o03LT_8RjpA52A")
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import os
import time

# Configure Gemini API
genai.configure(api_key="AIzaSyDM9xdKD9JDW_wu6Lp1gnCraUK3Ds-DPNc")

# Load UNet tumor classification model
model = load_model("unet_model.h5")
print("Model loaded successfully.")

# Hugging Face Medical NER API
API_URL = "https://api-inference.huggingface.co/models/blaze999/Medical-NER"
headers = {"Authorization": "Bearer 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"}
def query(payload, retries=5, wait_time=2):
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            data = response.json()
            if 'error' in data:
                print(f"Error: {data['error']}")
                if 'currently loading' in data['error'].lower():
                    print("Model is loading, retrying...")
                    time.sleep(wait_time)

                    continue
                    continue  
                return None
            return data
        except Exception as e:
            print(f"An error occurred during the API request: {e}")
            return None
    print("Max retries reached. Could not query the model.")
    return None

def create_entity_table(entities):
    if entities is None or not isinstance(entities, list):
        print("No entities to display or invalid format.")
        return None

    entity_data = {
        'Entity Type': [],
        'Entity Text': []
    }

    for entity in entities:
        entity_data['Entity Type'].append(entity.get('entity_group', 'Unknown'))
        entity_data['Entity Text'].append(entity.get('word', 'Unknown'))

    entity_table = pd.DataFrame(entity_data)
    print(entity_data)
    print("Entity table created successfully.")
    return entity_table

# Prompt template for summarizing prescriptions
base_prompt = """
As a medical assistant, you are asked to summarize the given patient prescription based on medical context.
Provide a clear, professional, single-paragraph summary highlighting key details and instructions.
"""

# Instantiate Gemini model once
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

def summarize_text(text):
    full_prompt = base_prompt + "\n\n" + text
    response = gemini_model.generate_content(
        full_prompt,
        generation_config=GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
            top_p=1
        )
    )
    return response.text

label_mapping = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary"
}

def get_label_from_prediction(predicted_index):
    label = label_mapping.get(predicted_index, "Unknown")
    print(f"Predicted label: {label}")
    return label

def visualize_prediction_from_image(img_path=None, img_url=None):
    try:
        if img_url:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
        elif img_path:
            img = Image.open(img_path)
        else:
            print("No image provided.")
            return None

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict tumor type
        prediction = model.predict(img_array)
        predicted_label_index = np.argmax(prediction, axis=1)[0]
        predicted_label = get_label_from_prediction(predicted_label_index)

        # Save the image with prediction result
        save_path = os.path.join("static", "uploads", f"prediction_result_{predicted_label}.png")
        img.save(save_path)
        print(f"Image saved at: {save_path}")
        return predicted_label, save_path
    except Exception as e:
        print(f"An error occurred during image prediction: {e}")
        return None

def output_report(image_path, custom_text):
    predicted_label, image_save_path = visualize_prediction_from_image(img_path=image_path)

    entities_output = query({"inputs": custom_text})
    entity_table = create_entity_table(entities_output)

    summarized_text = summarize_text(custom_text)

    report = {
        "Predicted Tumor Type": predicted_label if predicted_label else "Prediction failed",
        "Extracted Entities": entity_table.to_dict(orient='records') if entity_table is not None else [],
        "Summarized Text": summarized_text if summarized_text else "No summary available.",
        "Image Path": image_save_path if image_save_path else "No image path available"
    }

    return report
