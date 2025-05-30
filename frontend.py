import streamlit as st
from utils import get_response_from_api, get_vision_response, extract_text_and_images_with_mapping, save_text_image_mappings, find_relevant_images
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
        st.image(banner_image)
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
        # show_pdf = st.checkbox("View PDF", value=False)
        st.subheader("PDF Viewer")
        with st.expander("Click to view PDF", expanded=False):
            # Display PDF in iframe
            base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
          # Check if we need to process the PDF
        if 'processed_pdf' not in st.session_state or st.session_state.processed_pdf != uploaded_file.name:
            with st.spinner('Processing PDF...'):
                # Create a temporary file with a unique name
                temp_pdf_path = f"temp_{uploaded_file.name}"
                try:
                    # Save uploaded file temporarily
                    with open(temp_pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Extract text and images from PDF
                    text_image_mappings = extract_text_and_images_with_mapping(temp_pdf_path)
                    save_text_image_mappings(text_image_mappings)

                    # Read text content
                    pdf_document = fitz.open(temp_pdf_path)
                    text = ""
                    for page_num in range(len(pdf_document)):
                        page = pdf_document[page_num]
                        text += page.get_text()
                    pdf_document.close()

                    # Store in session state
                    st.session_state.text = text
                    st.session_state.text_image_mappings = text_image_mappings
                    st.session_state.processed_pdf = uploaded_file.name
                    
                    # Initialize FAISS vector store
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    st.session_state.vector_store = FAISS.from_texts([text], embeddings)

                finally:
                    # Clean up: Remove temporary file
                    if os.path.exists(temp_pdf_path):
                        try:
                            os.remove(temp_pdf_path)
                        except Exception as e:
                            print(f"Error removing temporary file: {e}")
        
        # Use stored values
        text = st.session_state.text

        # Show extracted images section
        if os.path.exists("text_image_mappings.json"):
            show_extracted_images = st.checkbox("View All Extracted Images", value=False)

            if show_extracted_images:
                st.subheader("All Extracted Images")
                with st.expander("Click to view images", expanded=False):
                    for mapping in st.session_state.text_image_mappings:
                        if os.path.exists(mapping['image_path']):
                            st.image(mapping['image_path'], 
                                   caption=f"Image from page {mapping['page_number']}", 
                                   use_column_width=True)


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
                
                # Get relevant images for the query
                relevant_images = find_relevant_images(user_input)
                
                # Show the response
                response = get_response_from_api(user_input, context_text)
                st.write("Answer:", response)
                
                # Show relevant images if any
                if relevant_images:
                    st.subheader("Most Relevant Image")
                    most_relevant_image = relevant_images[0]  # Get only the most similar image
                    if os.path.exists(most_relevant_image['image_path']):
                        st.image(most_relevant_image['image_path'], use_column_width=True)
                        st.write(f"Similarity Score: {most_relevant_image['similarity']:.2f}")
        # Show placeholder if no input
        if not user_input:
            st.info("Type your question and click Send to get an answer")

with tab2:
    st.subheader("Image Analysis")
    uploaded_image = st.file_uploader("Upload an image for analysis", type=["jpg", "png", "jpeg"])
    
    if uploaded_image:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
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
