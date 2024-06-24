import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from llama_index.core import download_loader, ServiceContext, VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core.response.pprint_utils import pprint_response

# Constants
STORAGE_PATH = "./vectorstore"
CONFIG = {
    "llama2": {
        "model_name": "NousResearch/Llama-2-7b-chat-hf",
        "embeddings_model_name": "all-MiniLM-L6-v2",
        "system_prompt": """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain
why instead of answering something not correct. If you don't know the answer
to a question, please don't share false information.

Your goal is to provide answers relating to the Wikipedia articles given.<</SYS>>
"""
    },
    "phi2": {
        "model_name": "microsoft/phi-2",
        "embeddings_model_name": "BAAI/bge-small-en-v1.5",
        "system_prompt": "You are a Q&A assistant. Your goal is to provide answers relating to the Wikipedia articles given"
    },
    "phi3": {
        "model_name": "microsoft/Phi-3-small-8k-instruct",
        "embeddings_model_name": "BAAI/bge-small-en-v1.5",
        "system_prompt": "You are a Q&A assistant. Your goal is to provide answers relating to the Wikipedia articles given"
    }
}

# Load environment variables
load_dotenv()
AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")  # Use environment variables

def get_model_config(model_choice):
    return CONFIG[model_choice]

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./model/', use_auth_token=AUTH_TOKEN, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir='./model/', use_auth_token=AUTH_TOKEN,
        torch_dtype=torch.bfloat16, rope_scaling={"type": "dynamic", "factor": 2},
        load_in_8bit=True, trust_remote_code=True
    )
    return tokenizer, model

def create_service_context(model_config):
    tokenizer, model = load_model(model_config["model_name"])
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=model_config["system_prompt"],
        query_wrapper_prompt=SimpleInputPrompt("{query_str}"),
        model=model,
        tokenizer=tokenizer
    )
    embeddings = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_config["embeddings_model_name"]))
    return ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model=embeddings)

def setup_index(service_context):
    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()
    documents = loader.load_data(pages=["2024 ICC Men's T20 World Cup"])
    index = VectorStoreIndex.from_documents(documents, embed_model=service_context.embed_model)
    index.storage_context.persist(persist_dir=STORAGE_PATH)
    return index

# Streamlit Web Interface
def main():
    st.title("Ask the Wiki On World Cups")
    model_choice = st.selectbox("Choose the model", list(CONFIG.keys()))
    model_config = get_model_config(model_choice)
    service_context = create_service_context(model_config)
    index = setup_index(service_context)
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, llm=service_context.llm)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me a question!"}]

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt)
                st.write(response.response)
                pprint_response(response, show_source=True)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
