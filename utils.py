import streamlit as st
import requests
import base64
from PIL import Image
import json
import os
import fitz
import io
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
        

# def handle_extracted_images(img_data, idx):    
#     # with st.expander(f"📄 Page {img_data['page']} - Image {idx + 1}"):
#         st.image(img_data['image'], 
#                 caption=f"Image {idx + 1} from page {img_data['page']}", 
#                 use_column_width=True)

def extract_text_and_images_with_mapping(pdf_path):
    """Extract text and images from PDF with their locations and create mappings"""
    text_image_mappings = []
    
    try:
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Get text blocks with their locations
            text_blocks = page.get_text("blocks")
            
            # Extract images
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    # Get image location on page
                    img_rect = page.get_image_bbox(img)
                    # Find nearby text blocks
                    nearby_text = ""
                    for block in text_blocks:
                        block_rect = fitz.Rect(block[:4])
                        # Check if text block is near the image (within 100 points)
                        # PyMuPDF Rect coordinates: y0 is top, y1 is bottom
                        if abs(block_rect.y0 - img_rect.y1) < 100 or abs(img_rect.y0 - block_rect.y1) < 100:
                            nearby_text += block[4] + " "
                    
                    # Create unique identifier for the image
                    image_id = f"page_{page_num + 1}_img_{img_index}"
                    
                    # Save image to file
                    image_filename = f"extracted_images/{image_id}.png"
                    os.makedirs("extracted_images", exist_ok=True)
                    image.save(image_filename)
                    image.close()  # Close the image after saving
                    clean_text = re.sub(r'<image:[^>]+>', '', nearby_text)  # Remove image tag
                    clean_text = re.sub(r'\s+', ' ', clean_text.strip().replace('\n', ' ')) 
                    # Store mapping
                    mapping = {
                        "image_id": image_id,
                        "image_path": image_filename,
                        "page_number": page_num + 1,
                        "nearby_text": clean_text,
                        "location": {
                            "top": img_rect.y0,
                            "bottom": img_rect.y1,
                            "left": img_rect.x0,
                            "right": img_rect.x1
                        }
                    }
                    text_image_mappings.append(mapping)
                except Exception as e:
                    print(f"Error processing image {img_index} on page {page_num + 1}: {e}")
                    continue
    finally:
        # Ensure PDF document is closed
        if 'pdf_document' in locals():
            pdf_document.close()
    
    return text_image_mappings

def save_text_image_mappings(mappings, output_file="text_image_mappings.json"):
    """Save text-image mappings to a JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=4)

def find_relevant_images(query_text, mappings_file="text_image_mappings.json", similarity_threshold=0.5):
    """Find images relevant to the query text using semantic similarity"""
    try:
        # Load the mappings
        with open(mappings_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        # Initialize the sentence transformer model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Encode the query text
        query_embedding = model.encode([query_text])[0]
        
        relevant_images = []
        for mapping in mappings:
            # Encode the nearby text
            text_embedding = model.encode([mapping['nearby_text']])[0]
            
            # Calculate similarity
            similarity = cosine_similarity([query_embedding], [text_embedding])[0][0]
            
            if similarity > similarity_threshold:
                relevant_images.append({
                    'image_path': mapping['image_path'],
                    'similarity': float(similarity),
                    'nearby_text': mapping['nearby_text']
                })
        
        # Sort by similarity score
        relevant_images.sort(key=lambda x: x['similarity'], reverse=True)
        return relevant_images
    except Exception as e:
        print(f"Error finding relevant images: {str(e)}")
        return []
