---
title: Mycheckspace
emoji: ⚡
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Basic LLM Chatbot with LangChain and HuggingFace

This project is a simple chatbot application built using LangChain and HuggingFace Transformers. It provides a conversational interface where users can interact with a large language model in real-time. Features include a chat interface with Gradio, integration with HuggingFace models, context-aware chat history, and easily replaceable backend models such as Flan-T5 or Google Gemma.

## Requirements
- Python 3.10+
- pip packages: transformers, langchain-huggingface, gradio, torch (with GPU support if available)

## Setup
1. Clone the repository:  
   `git clone <your-repository-url>`
2. Navigate into the project folder:  
   `cd basic-ai-chatbot`
3. Create and activate a virtual environment:  
   Linux/Mac: `python -m venv .venv && source .venv/bin/activate`  
   Windows: `.venv\Scripts\activate`
4. Install dependencies:  
   `pip install -r requirements.txt`
5. Set your HuggingFace token as an environment variable:  
   Linux/Mac: `export HF_TOKEN=<your_huggingface_token>`  
   Windows: `set HF_TOKEN=<your_huggingface_token>`

## Usage
Run the chatbot application:  
`python app.py`  
Open the URL displayed in the terminal to start chatting.

## Configuration
- `model_name`: specify the HuggingFace model to use
- `max_new_tokens`: limit the length of generated responses
- In-memory chat history maintains context across the conversation

## Notes
- Large models like `google/gemma-2b` may respond slowly on free HuggingFace Spaces due to compute limitations.
- Never commit your HuggingFace token to public repositories. Use environment variables or secrets management.
- This project is licensed under the MIT License.