# MedicalBot - AI-powered Medical Report Generator

This is a Flask-based application designed to help doctors and healthcare professionals upload a patient's MRI image and prescription for disease prediction and Named Entity Recognition (NER) from the prescription text. The app utilizes a UNet architecture for disease prediction from medical imaging and performs automatic summarization of the prescription. 

## Features

- **Image Upload**: Upload MRI images of patients to detect brain tumor types using a trained UNet model.
- **Prescription Text Analysis**: Upload prescription text to extract medical entities using Named Entity Recognition (NER).
- **Disease Prediction**: Classifies brain tumor types into four categories: glioma, meningioma, no tumor, and pituitary tumor.
- **Summarization**: Automatically generates a concise summary of the patient's prescription text.

## How it Works

1. **Upload MRI Image**: The user uploads an MRI image, and the system uses the UNet model to predict the presence and type of brain tumor.
2. **Upload Prescription Text**: Users can also input a patient's prescription text for analysis.
3. **Named Entity Recognition (NER)**: The prescription text is processed to extract key medical entities (such as drug names, diseases, and symptoms).
4. **Summarization**: The system summarizes the prescription text, providing a concise report based on the provided prescription.

## Model Information

- **Disease Prediction**: The disease prediction model uses a UNet architecture trained on MRI brain tumor datasets. It categorizes brain tumors into four types.
- **NER**: NER is performed using a pre-trained medical model hosted on Hugging Face. The model identifies relevant medical entities in the text.
- **Summarization**: A summarization model is employed to condense the prescription text into a shorter, medically relevant report.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/khushie03/MedicalBot.git
   cd MedicalBot
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Set up API Keys:**
   - Get your Hugging Face API key and Gemini API key.
   - Update them in `main.py` as:
     ```python
     genai.configure(api_key="Your Gemini API Key")
     headers = {"Authorization": "Bearer Your Hugging Face API Key"}
     ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the application:**
   Visit `http://127.0.0.1:5001/` in your web browser.

## Usage

1. Upload an MRI image on the main page.
2. Enter or upload a patient's prescription.
3. Click **Submit** to generate the disease prediction, extract medical entities, and view the summarized prescription.

## Directory Structure

```
├── app.py                  # Main Flask app
├── main.py                 # Logic for model predictions, NER, and summarization
├── templates/
│   ├── index.html          # Upload form for image and prescription
│   ├── main.html           # Main page template
│   └── result.html         # Display results (NER, Prediction, and Summary)
├── static/
│   └── uploads/            # Uploaded images
├── requirements.txt        # Dependencies for the project
└── README.md               # Project documentation
```

## Dependencies

- Flask
- Transformers (Hugging Face)
- Google Generative AI
- TensorFlow
- Pandas
- Pillow
- Numpy
