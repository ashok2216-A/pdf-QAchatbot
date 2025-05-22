import streamlit as st
import requests
import base64
from PIL import Image

# FastAPI endpoint URLs
API_URL = "http://localhost:8000"

def get_response_from_api(prompt, context=None):
    try:
        if 'mistral_api_key' not in st.session_state:
            return "Please enter your Mistral API key in the sidebar settings"
            
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "prompt": prompt,
                "context": context,
                "api_key": st.session_state['mistral_api_key']
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"Error: {str(e)}"

def get_vision_response(image_path, prompt="What's in this image?"):
    try:
        if 'mistral_api_key' not in st.session_state:
            return "Please enter your Mistral API key in the sidebar settings"
            
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = requests.post(
            f"{API_URL}/vision",
            json={
                "image": base64_image,
                "prompt": prompt,
                "api_key": st.session_state['mistral_api_key']
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"Error: {str(e)}"

def handle_extracted_images(img_data, idx):    
    with st.expander(f"📄 Page {img_data['page']} - Image {idx + 1}"):
        st.image(img_data['image'], 
                caption=f"Image {idx + 1} from page {img_data['page']}", 
                use_container_width=True)
