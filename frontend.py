import streamlit as st
from utils import get_response_from_api, get_vision_response, handle_extracted_images
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import fitz  # For PDF processing
import os
from PIL import Image
import io
import base64


# Initialize Streamlit app
st.title("PDF Chatbot with Mistral AI")
        
# Sidebar
with st.sidebar:
    # Load and display the banner image if it exists
    try:
        banner_image = Image.open("banner.PNG")
        st.image(banner_image, use_container_width=True)
    except (FileNotFoundError, IOError):
        st.warning("Banner image not found")
    
    st.title("Settings")
    
    # Mistral API Key input
    with st.expander("⚙️ API Configuration", expanded=True):
        api_key = st.text_input("Enter Mistral API Key:", type="password")
        if api_key:
            st.session_state['mistral_api_key'] = api_key
            st.success("✅ API Key saved!")
        elif 'mistral_api_key' not in st.session_state:
            st.warning("Please enter your Mistral API key")
    
    
    # File uploader for PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # PDF viewer toggle
        show_pdf = st.checkbox("View PDF", value=False)
        
        if show_pdf:
            # Display PDF in iframe
            base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
          # Check if we need to process the PDF
        if 'processed_pdf' not in st.session_state or st.session_state.processed_pdf != uploaded_file.name:
            with st.spinner('Processing PDF...'):
                # Save uploaded file temporarily and process
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Extract text and images from PDF
                pdf_document = fitz.open("temp.pdf")
                text = ""
                images = []

                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    text += page.get_text()

                    # Extract images
                    for img_index, img in enumerate(page.get_images(full=True)):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes))
                        images.append({"image": image, "page": page_num + 1})

                pdf_document.close()

                # Store in session state
                st.session_state.text = text
                st.session_state.images = images
                st.session_state.processed_pdf = uploaded_file.name
                
                # Initialize FAISS vector store
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.vector_store = FAISS.from_texts([text], embeddings)
        
        # Use stored values
        text = st.session_state.text
        images = st.session_state.images

        # Show extracted images section
        if images:
            # st.markdown("---")  # Add separator
            show_extracted_images = st.checkbox("View Extracted Images", value=False)
            
            if show_extracted_images:
                st.subheader("Extracted Images")
                for idx, img_data in enumerate(images):
                    handle_extracted_images(img_data, idx)


# Create tabs in main content area
st.markdown("Please Upload a PDF and Start Interacting with it!")
tab1, tab2 = st.tabs(["PDF Text Analysis", "PDF Image Analysis"])

# clean up temporary files
if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")

with tab1:
    if uploaded_file is not None:
        # Main content area
        st.subheader("Extracted Text")
        st.text_area("", text, height=300)        # Use stored vector store
        vector_store = st.session_state.vector_store
        
        # Chat interface
        st.subheader("Chat with the PDF")

        user_input = st.text_input("Ask a question about the PDF:", key="chat_input")
        send_button = st.button("Send", type="primary")

        # Only process when send button is clicked
        if send_button and user_input:
            with st.spinner('Getting response...'):
                context = vector_store.similarity_search(user_input, k=1)
                context_text = "\n".join([doc.page_content for doc in context])
                
                response = get_response_from_api(user_input, context_text)
                st.write("Answer:", response)
        
        # Show placeholder if no input
        if not user_input:
            st.info("Type your question and click Send to get an answer")

with tab2:
    st.subheader("Image Analysis")
    uploaded_image = st.file_uploader("Upload an image for analysis", type=["jpg", "png", "jpeg"])
    
    if uploaded_image:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            analyze_button = st.button("Analyze Image")
        
        with col2:
            if analyze_button:
                # Save temporary image
                with open("temp_image.jpg", "wb") as f:
                    f.write(uploaded_image.getbuffer())
                
                # Get vision analysis
                vision_response = get_vision_response("temp_image.jpg")
                st.write("Analysis:", vision_response)
                
                # Clean up
                os.remove("temp_image.jpg")