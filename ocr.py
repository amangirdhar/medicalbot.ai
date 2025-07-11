import easyocr

# Create an EasyOCR reader (English language by default)
reader = easyocr.Reader(['en'])

# Run OCR on the image
results = reader.readtext(r"C:\Users\HP\Downloads\OIP (5).jpg")

# Print results
for (bbox, text, confidence) in results:
    print(f"Detected Text: {text}, Confidence: {confidence:.2f}")
