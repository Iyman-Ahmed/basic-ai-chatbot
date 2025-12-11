import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import os
import warnings
# Hugging Face model and token
model_name = "google/gemma-2b"
HF_TOKEN = os.getenv("HF_token")  # Store your token as environment variable

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    use_auth_token=HF_TOKEN
)

# Pipeline for text generation
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
)

# Conversation history
history = []

# Chat function
def chat_with_model(message, chat_history):
    global history
    if not isinstance(chat_history, list):
        chat_history = []

    # Build prompt with conversation history
    conversation = ""
    for h in history:
        conversation += f"User: {h['user']}\nAssistant: {h['assistant']}\n"
    conversation += f"User: {message}\nAssistant:"

    # Generate response
    response_full = generator(conversation, do_sample=True, temperature=0.7)[0]['generated_text']
    response = response_full.split("Assistant:")[-1].strip()  # Get only latest assistant reply

    # Update history
    history.append({"user": message, "assistant": response})
    chat_history.append({"role":"user","content":message})
    chat_history.append({"role":"assistant","content":response})

    return chat_history, chat_history

# CSS for styling
css = """
.gr-chatbot {
    max-height: 400px;
}
.gr-textbox textarea {
    max-height: 50px;
    font-size: 14px;
    padding: 6px 10px;
    border-radius: 8px;
}
"""

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("### Chatbot with Google Gemma-2B")
    chatbot = gr.Chatbot()
    state = gr.State([])
    user_input = gr.Textbox(placeholder="Type your message here...", lines=1)
    send_button = gr.Button("Send")

    send_button.click(
        chat_with_model,
        inputs=[user_input, state],
        outputs=[chatbot, state]
    )

demo.launch(share = False, debug = True, css=css, prevent_thread_lock=False)
warnings.filterwarnings("ignore", category=ResponseWarning)