import google.generativeai as genai
from serpapi import GoogleSearch

genai.configure(api_key="AIzaSyDM9xdKD9JDW_wu6Lp1gnCraUK3Ds-DPNc")

base_prompt = """
Based on the given query and conversation history, 
Basically you need to act as a medical chatbot and answer to user queries . Provide answer to their 
"""
def generate_message(query, conversation_history):
    conversation_context = "\n".join(f"User: {item['user']}\nBot: {item['bot']}" for item in conversation_history)
    full_prompt = f"{base_prompt}\n{conversation_context}\nUser: {query}\nBot:"
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(full_prompt)
    return response.text

