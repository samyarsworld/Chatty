import os
import pathlib
import requests
import streamlit as st
from typing import List

from langsam.libs.constants import ALLOWED_FILE_TYPES
from langsam.chatty import Chatty
from langsam.libs.message import UserMessage


def handle_file_upload() -> None:
    """
    Handle uploaded files.
    """

    # Display title of the webpage
    st.title("Your friendly personal chatbot says Hello 😁\n Upload your document and ask questions!")

    # Handle uploaded file
    uploaded_file = st.file_uploader(
        label=(
            f"Choose a {', '.join(ALLOWED_FILE_TYPES[:-1]).upper()}, or "
            f"{ALLOWED_FILE_TYPES[-1].upper()} file"
        ),
        type=ALLOWED_FILE_TYPES
    )
    if uploaded_file:
        # Determine the file type
        file_type = pathlib.Path(uploaded_file.name).suffix.replace(".", "")

        if file_type:            
            # Create temporary directory
            if not os.path.exists("temp"):
                os.makedirs("temp")

            # Store the file in io.BytesIO format to temporarily store in memory
            file_path = os.path.join("temp", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state["file_path"] = file_path
            st.session_state["file_type"] = file_type
            st.success("File uploaded successfully!😊\n I'll display available LLMs shortly..")

            # Display the available language models from Ollama
            # and store the selected model for inference
            models = get_models("http://ollama:11434")
            if models:
                selected_model = st.selectbox("Select Model", models)
                st.session_state['selected_model'] = selected_model            
                st.button(
                    "Ready to Chat!",
                    on_click=lambda: st.session_state.update({"page": "chat_interface_page"})
                )
        else:
            st.error(
                f"Unsupported file type. Please upload a "
                f"{', '.join(ALLOWED_FILE_TYPES[:-1]).upper()}, or "
                f"{ALLOWED_FILE_TYPES[-1].upper()} file."
            )



def chat_interface() -> None:
    """
    Main chat interface - invoked after a file has been uploaded.
    """

    # Display title of the webpage
    st.title("Document Buddy - Chat with Document Data")

    # Get file from session state
    file_path = st.session_state.get("file_path")
    file_type = st.session_state.get("file_type")
    if not file_path or not os.path.exists(file_path):
        st.error("File missing! Please upload a file.")
        return

    # Get chatbot instance from session state
    if "chatbot_instance" not in st.session_state:
        st.session_state["chatbot_instance"] = Chatty(
            file_path=file_path,
            file_type=file_type,
            embedding_model="hkunlp/instructor-large",
            device='cuda'
        )
    
    # Display user query input
    user_input = st.text_input("Ask a question:")
    if user_input and st.button("Send"):
        with st.spinner("Thinking..."):
            top_result = st.session_state["chatbot_instance"].chat(user_input)

            # Display top answers
            if top_result:
                st.markdown("**Top Answer:**")
                st.markdown(f"> {top_result['answer']}")
            else:
                st.write("No answer found.")

            # Display chat history
            st.markdown("**Chat History:**")
            for message in st.session_state["chatbot_instance"].conversation_history:
                prefix = "*You:* " if isinstance(message, UserMessage) else "*AI:* "
                st.markdown(f"{prefix}{message.content}")


def get_models(base_url: str) -> List[str]:
    try:       
        # Request model and save the model data
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()
        models_data = response.json()

        # Extract model names
        models = [model['name'] for model in models_data.get('models', [])]
        return models
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get models from Ollama: {e}")
        return []
    
if __name__ == '__main__':
    if "page" not in st.session_state:
        st.session_state["page"] = 'file_upload_page'

    if st.session_state["page"] == 'file_upload_page':
        handle_file_upload()
    elif st.session_state["page"] == 'chat_interface_page':
        chat_interface()