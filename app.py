import os
# Suppress warnings and OpenMP errors BEFORE importing other libraries
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import tempfile
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Import existing models logic
from MLM.mlm import get_chain as mlm_get_chain
from MoE.moe import get_chain as moe_get_chain
from SLM.slm import get_chain as slm_get_chain
from VLM.vlm import get_vlm_response
from SAM.sam import get_generator as sam_get_generator, generate_masks_from_image
from groq import Groq


# Load env variables
load_dotenv()


# --- Config & Initialization ---
st.set_page_config(page_title="AI Models Interface", page_icon="🤖", layout="wide")

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Model Selector")
    model_choice = st.radio(
        "Choose a model to interact with:",
        ("MLM", "MoE", "SLM", "VLM", "SAM")
    )
    



# --- Helper Functions for Chat ---
def clear_chat_history():
    st.session_state.messages = []

# Initialize chat history if not present or if model changed
if "messages" not in st.session_state or "current_model" not in st.session_state or st.session_state.current_model != model_choice:
    st.session_state.messages = []
    st.session_state.current_model = model_choice


# --- Shared Chat UI Function for MLM, MoE, SLM ---
def chat_interface(model_display_name: str, get_chain_func):
    st.title(f"💬 {model_display_name}")
    st.button("Clear Chat", on_click=clear_chat_history)
    
    if not st.session_state.groq_api_key:
        st.warning("Please set the GROQ_API_KEY environment variable to continue.")
        return

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # Re-use the existing chain initialization logic from the script
            chain = get_chain_func(st.session_state.groq_api_key)
        except Exception as e:
            st.error(f"Error initializing chain: {e}")
            return

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chain.invoke({"input": prompt})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error parsing response: {e}")


# --- Main Logic ---

if model_choice == "MLM":
    chat_interface("Medium Language Model", mlm_get_chain)

elif model_choice == "MoE":
    chat_interface("Mixture of Experts", moe_get_chain)

elif model_choice == "SLM":
    chat_interface("Small Language Model", slm_get_chain)

elif model_choice == "VLM":
    st.title(f"👁️ Vision Language Model")
    st.write("Upload an image and ask a question about it.")
    
    if not st.session_state.groq_api_key:
        st.warning("Please set the GROQ_API_KEY environment variable to continue.")
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption='Uploaded Image', use_container_width=True)
                
        with col2:
            question = st.text_area("Ask a question about the image:", value="Describe this image in detail.")
            
            if st.button("Analyze Image"):
                if uploaded_file is None:
                    st.error("Please upload an image first.")
                elif not question:
                    st.error("Please ask a question.")
                else:
                    with st.spinner("Analyzing using existing VLM logic..."):
                        client = Groq(api_key=st.session_state.groq_api_key)
                        
                        # Save uploaded file to temp file, as VLM logic expects a file path
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                            image.save(tmp_file.name)
                            tmp_path = tmp_file.name
                        
                        try:
                            # Re-use the existing get_vlm_response logic
                            result = get_vlm_response(client, tmp_path, question)
                            st.markdown("### Response:")
                            st.write(result)
                        except Exception as e:
                            st.error(f"Error: {e}")
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)

elif model_choice == "SAM":
    st.title(f"🖼️ Segment Anything Model")
    st.write("Upload an image to generate segmentation masks.")
    
    @st.cache_resource
    def load_sam_model_cached():
        # Re-use the existing model initialization
        return sam_get_generator()
        
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp", "bmp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button("Generate Masks"):
            with st.spinner("Loading Model & Generating Masks using existing logic... This might take a moment."):
                try:
                    generator = load_sam_model_cached()
                    # Re-use the extraction method
                    annotated_img, detected_classes = generate_masks_from_image(generator, image)
                    
                    if detected_classes:
                        formatted_classes = ", ".join(detected_classes)
                        st.success(f"Successfully segmented the following objects: **{formatted_classes}**")
                    else:
                        st.info("Successfully generated segmentation masks, but no specific named objects were recognized.")
                        
                    st.image(annotated_img, caption="Segmented Image", use_container_width=True)
                    
                except ImportError as e:
                    st.error(f"Failed to load requirements. {e}")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
