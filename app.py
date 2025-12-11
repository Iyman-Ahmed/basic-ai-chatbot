import gradio as gr
from transformers import AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM
import os

# Load model
model_name = "google/gemma-2b"

HF_t=  os.getenv("HF_token") 
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_t)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    use_auth_token=HF_t,
)


pipe_line = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
)

llm = HuggingFacePipeline(pipeline=pipe_line)

def chat_with_model(message, history):
    if not isinstance(history, list):
        history = []

    response = llm.invoke(message)
    #store user message
    history.append({'role':'user', 'content': message})
    #store model response
    history.append({'role':'assistant', 'content': response})
    return history, history


css = """
.chatbot-box {
    background-color: #DCF8C6;
    color: #000000;
    size: 150px;
    }
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("### Basic LLM Chatbot with LangChain + HuggingFace")
    chatbot = gr.Chatbot(elem_classes = "chatbot-box")
    state = gr.State([])  
    user_input = gr.Textbox(placeholder="Type your message...", label="Your Message",elem_id="user-input",elem_classes="user-input-box")
    send_button = gr.Button("Send")

    send_button.click(
        chat_with_model,
        inputs=[user_input, state],
        outputs=[chatbot, state]
    )

demo.launch(css = css)