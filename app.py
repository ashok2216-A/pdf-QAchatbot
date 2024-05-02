import PyPDF2
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-3 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-125M")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfFileReader(f)
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
    return text

def generate_response(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def main():
    st.title("PDF Chatbot")

    # File upload
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.text_area("PDF Content", pdf_text)

    user_input = st.text_input("You:", "")
    if st.button("Send"):
        response = generate_response(user_input)
        st.text_area("Chatbot:", response)

if __name__ == "__main__":
    main()
