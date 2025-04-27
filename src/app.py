import streamlit as st
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from config.config_loader import ConfigLoader # CUSTOM CLASS
from models.model_loader import ModelLoader # CUSTOM CLASS
from utils.constants import TASK_OPTIONS #, DEFAULT_MODEL_PARAMS # CUSTOM CLASS/PARAMS
from utils.paths import PathManager # CUSTOM CLASS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from IPython.display import Image, display
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
# from ollama_functions_custom import OllamaFunctions
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from typing import (
    Annotated,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
)
import base64
import contextlib
import faiss
import gc
import getpass
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import pyarrow.feather as feather
import pyarrow.parquet as pq
import re
import sys
import tempfile
import textwrap
import threading
import time
import torch
import traceback
import yaml
# from sql_utility import list_tables_tool
# from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool, InfoSQLDatabaseTool, ListSQLDatabaseTool, QuerySQLCheckerTool
# from tqdm.autonotebook import tqdm, trange
# from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama
# from langchain_community.chat_models import ChatLlamaCpp
# from langchain_core.runnables import RunnablePassthrough

# Define global variables at the module level
# os.environ["POLARS_MAX_THREADS"] = "1"

# PAUSE
Pause_productivity_Analysis = True
model_embeddings, llm_agent = None, None

# LLMs

if 'second_task' not in st.session_state:
    st.session_state.second_task = False


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """
    Load models
    """
    # --- Progress Indicators ---
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # --- Load LLMs ---
    progress_text.text("Loading LLMs...")
    print("Loading LLMs...") # Debug print
  
    gc.collect()
    torch.cuda.empty_cache()

    progress_bar.progress(25)
    
    print('---')
    progress_bar.progress(50)

    print('---')
    progress_bar.progress(100)
    
    # --- Cleanup ---
    progress_text.empty()
    progress_bar.empty()

    print("--- Finished load_models ---") # Debug print
    
    model_embeddings, llm_agent = None, None
    return model_embeddings, llm_agent

def initialize_session_state():
    if 'stop_streaming' not in st.session_state:
        st.session_state.stop_streaming = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "statistics" not in st.session_state:
        st.session_state.statistics = []
    if "context" not in st.session_state:
        st.session_state.context = ""

def setup_sidebar():
    with open("./assets/logo.png", "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    with open("./src/static/about.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    with open("./src/static/sidebar.css") as f:
        st.sidebar.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    logo_and_text = f"""
    <div class='logo-container'>
        <img src="data:image/png;base64,{base64_image}" class="rotating-logo">
        <div class='sidebar-text'>Synapse</div>
    </div>
    """
    st.sidebar.markdown(logo_and_text, unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-subtext'>Explore AI Agents Workflows</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr class='sidebar-divider'></hr>", unsafe_allow_html=True)

    return st.sidebar.selectbox(options = TASK_OPTIONS, label='', label_visibility="collapsed")

def set_about_section():
    with st.sidebar.expander("About"):
        with open("./src/static/about.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        with open("./src/static/about.html") as f:
            st.markdown(f"{f.read()}", unsafe_allow_html=True)
    
    with open("./src/static/note_ai.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    st.sidebar.markdown("""<p class="note-ai">Occasionally, AI workflows may produce unexpected results.</p>""", unsafe_allow_html=True)

def main():
    try:
        st.set_page_config(
            page_title="productivity Chatbot | Acrivon Phosphoproteomics",
            page_icon="./assets/logo.png",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except:
        pass

    if 'inner_auto' not in st.session_state:
        st.session_state.inner_auto = False

    print("--- Starting main function ---") 
    load_success = False
    try:
        print("Calling load_models_dataframe...") 
        model_embeddings, llm_agent = load_models() # Ensure globals are set
        load_success = True
    except Exception as e:
        st.error(f"Fatal error during resource loading: {e}")
        st.exception(e) # Show full traceback in Streamlit for detailed debugging
        print(f"Exception during load_models_dataframe call: {e}") # Debug
        st.stop() # Stop the Streamlit app execution

    with open("./app/style.css") as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

    # Read and encode the logo image
    with open("./assets/logo.png", "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

    path_manager = PathManager()
    
    try:        
        initialize_session_state()      
        selected_task = setup_sidebar()

        with open("./src/static/mian_area.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)     
        set_about_section()

        if 'first_question' not in st.session_state or len(st.session_state.first_question) == 0:
            with open("./src/static/main_chat_input.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 16, 1])
            files_exist = False
            with col2:
                st.markdown("""
                    <div class="empty-chat-container">
                        <h1 class="welcome-text">Welcome to Synapse Workflows</h1>
                        <p class="welcome-subtext">
                        Interact with powerful AI Agent Workflows using natural language. Try running <br>
                        <code>Smart Search Agent</code>, <code>Productivity Assistant</code>, or <code>Data Analysis Agent</code>.<br>
                        Ready to explore the world of intelligent agents?
                        </p><br>
                    </div>
                    """, unsafe_allow_html=True)
                
                if selected_task == "Smart Search Agent": 
                    total_input = st.chat_input(
                        key="initial_query",
                        placeholder="Search or Ask anything...",
                    )   

                    user_input = total_input
                elif selected_task == "Auto Selection": 
                    total_input = st.chat_input(
                        key="initial_query",
                        accept_file  = "multiple",
                        file_type = ["pdf", "jpg", "jpeg", "png", "doc", "docx"],
                        placeholder = (
                            "Ask anything...\n\nNot sure which agent you need? "
                            "Type your question or upload files, and Synapse will auto-select the best workflow for you!"
                        ),
                    )   

                    if total_input:
                        if hasattr(total_input, 'text') and total_input.text:
                            user_input = total_input.text
                        if hasattr(total_input, 'files') and total_input.files:
                            files_collected = total_input['files']
                            files_exist = True
                else:
                    total_input = st.chat_input(
                        key="initial_query", 
                        accept_file  = "multiple",
                        file_type = ["pdf", "jpg", "jpeg", "png", "doc", "docx"],
                        placeholder="Ask Anything...",
                    )

                    if total_input:
                        if hasattr(total_input, 'text') and total_input.text:
                            user_input = total_input.text
                        if hasattr(total_input, 'files') and total_input.files:
                            files_collected = total_input['files']
                            files_exist = True
                         
                if total_input:
                    if 'first_question' not in st.session_state:
                        st.session_state.first_question = []

                    if not files_exist:
                        st.session_state.first_question.append({"role": "initializer", "content": user_input, "file": None})
                    else:
                        st.session_state.first_question.append({"role": "initializer", "content": user_input, "file": files_collected})

                    st.rerun()
                
                # Load custom HTML layout
                with open("./src/static/static_feature_card.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
                    st.markdown(html_content, unsafe_allow_html=True)
                
                from streamlit_extras.bottom_container import bottom

                with bottom():
                    with open("./src/static/links.html", "r", encoding="utf-8") as f:
                        html_content = f.read()
                        st.markdown(
                            f"""
                            <div style="max-height: 70px; overflow-y: auto;">
                                {html_content}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        else:
            task = selected_task
            if selected_task=='Auto Selection' and 'global_task' not in st.session_state:
                st.session_state.global_task = "Auto Selection"
            elif 'global_task' in st.session_state and st.session_state.global_task == "Inner Auto Selection":
                pass
           
            if 'stop_streaming' not in st.session_state:
                st.session_state.stop_streaming = False

            if "messages" not in st.session_state:
                st.session_state.messages = []
                st.session_state.context = ""

            from PIL import Image
            def process_avatar(image_path, output_size=200):
                try:
                    img = Image.open(image_path)
                    if img.mode != "RGBA":
                        img = img.convert("RGBA")
                    
                    # Create a square canvas with transparency
                    canvas = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 0))
                    
                    # Maintain aspect ratio
                    width, height = img.size
                    ratio = min(output_size / width, output_size / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    
                    # Resize the image while maintaining aspect ratio
                    resized = img.resize(new_size, Image.LANCZOS)
                    
                    # Center the resized image on the canvas
                    x = (output_size - new_size[0]) // 2
                    y = (output_size - new_size[1]) // 2
                    canvas.paste(resized, (x, y), resized)  # Use the resized image as a mask for transparency
                    
                    processed_path = os.path.join("./assets", "processed_" + os.path.basename(image_path))
                    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
                    canvas.save(processed_path)
                    
                    return processed_path

                except FileNotFoundError:
                    print(f"Avatar image not found at: {image_path}")
                    return None
                except Exception as e:
                    print(f"An error occurred: {e}")

            assistant_avatar = process_avatar("./assets/logo.png")
            user_avatar = "./assets/logo.png"

            # color_scheme = {
            #     'assistant-title': '#CC7900',
            #     'human_heading': '#FF9800',  # Green for Human Message
            #     'ai_heading': '#2196F3',  # Blue for AI Message
            #     'tool_heading': '#FF9800',  # Orange for Tool Message
            #     'expander_title': '#673AB7',  # Deep Purple for Expander
            #     'cyan-title-start': '#22808e',  # Use magenta
            #     'cyan-title-start-code': '#e83e8c',  #
            # }
            # CSS to style the headings .stop-button {{color: 'tool_heading']}; cursor: pointer; margin-left: 10px;}}

            with open("./src/static/secondpage.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

            st.markdown("""
            <script>
                // Utility function to debounce frequent events
                function debounce(func, wait, immediate) {
                    let timeout;
                    return function() {
                        const context = this, args = arguments;
                        const later = function() {
                            timeout = null;
                            if (!immediate) func.apply(context, args);
                        };
                        const callNow = immediate && !timeout;
                        clearTimeout(timeout);
                        timeout = setTimeout(later, wait);
                        if (callNow) func.apply(context, args);
                    };
                }

                // Function to position the stop button
                function positionStopButton() {
                    const chatInput = document.querySelector('[data-testid="stChatInput"]');
                    const stopButton = document.getElementById('stop-button');
                    
                    if (chatInput && stopButton) {
                        const rect = chatInput.getBoundingClientRect();
                        const buttonWidth = stopButton.offsetWidth;
                        
                        // Position the button to the left of the chat input
                        stopButton.style.top = `${rect.top}px`;
                        stopButton.style.left = `${rect.left - buttonWidth - 30}px`; // 30px gap
                        stopButton.style.bottom = `${rect.bottom - 10}px`;
                        
                        // Make sure button is visible
                        stopButton.style.display = 'block';
                    }
                }

                // Create and append the stop button
                function createStopButton() {
                    const button = document.createElement('button');
                    button.id = 'stop-button';
                    button.className = 'stop-button';
                    button.textContent = 'Stop';
                    button.onclick = function() {
                        // Add your stop functionality here
                        console.log('Stop button clicked');
                        this.disabled = true;
                        // You can emit a custom event that Streamlit can listen to
                        const event = new CustomEvent('stopGeneration', { detail: { stopped: true } });
                        window.dispatchEvent(event);
                    };
                    document.body.appendChild(button);
                }

                // Initialize everything when the DOM is loaded
                function initialize() {
                    createStopButton();
                    positionStopButton();
                    
                    // Set up resize handler with debouncing
                    const debouncedPosition = debounce(positionStopButton, 100);
                    window.addEventListener('resize', debouncedPosition);
                    
                    // Reposition on any Streamlit rerun
                    new MutationObserver((mutations) => {
                        for (const mutation of mutations) {
                            if (mutation.type === 'childList') {
                                positionStopButton();
                            }
                        }
                    }).observe(document.body, { childList: true, subtree: true });
                }

                // Wait for DOM to be ready
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', initialize);
                } else {
                    initialize();
                }
            </script>
            """, unsafe_allow_html=True)


            # Process uploaded PDFs
            def process_pdf(pdf_list, chunk_size=1000, chunk_overlap=100):
                total_chunks = 0
                page_content_lengths = []
                with st.spinner("📚 Processing your PDFs... Please wait while we extract all that knowledge!"):
                    for file in pdf_list:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                            tmp.write(file.getvalue())
                            time.sleep(1)
                            tmp_path = tmp.name

                        loader = PyPDFLoader(tmp_path)
                        documents = loader.load()

                        os.unlink(tmp_path)

                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        chunks = text_splitter.split_documents(documents)
                        page_content_lengths.extend([len(chunk.page_content) for chunk in chunks])

                        embeddings = model_pdf #HuggingFaceEmbeddings(model_name=str(path_manager.model_paths["EMBEDDINGS_qwen"]))
                        # print(">>>>", str(path_manager.model_paths["EMBEDDINGS_qwen"]))

                        # Create or update vector store
                        if st.session_state.vectorstore:
                            st.session_state.vectorstore.add_documents(chunks)
                        else:
                            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                        
                        total_chunks += len(chunks)
                
                fun_facts = [
                    "Fun fact: The average business document contains only 20% truly unique content.",
                    "AI Tidbit: These embeddings help your AI understand meaning, not just words!",
                    "Tech note: Vector databases are how AIs can 'remember' your documents.",
                    "Curious? Each 'chunk' is like a small puzzle piece of your document's knowledge!"
                ]
                import random
                
                st.session_state.pdf_processed = {
                    "num_chunks": total_chunks,
                    "fun_fact": random.choice(fun_facts),
                    "num_documents": len(pdf_list),
                    "chunk_sizes": page_content_lengths if total_chunks > 0 else []
                }
                
                return total_chunks

            def supervisor_node(user_input: str):
                class Router(BaseModel):
                    """Worker to route to next."""
                    next: Literal["smart_search", "productivity_assistant", "data_analysis_agent"]
                
                members = ["smart_search", "productivity_assistant", "data_analysis_agent"]
                
                system_prompt = f"""You are a supervisor tasked with managing a conversation between the following workers: {members}.
    Given the user request, determine which worker should handle it next.

    Return ONLY a JSON object with a single field 'next' that must be one of: "smart_search" or "productivity_assistant" or "data_analysis_agent".

    Example response format:
    {{{{
        "next": "productivity_assistant"
    }}}}

    Routing criteria with detailed examples:
    - Route to "smart_search" when the query involves finding information, answering factual questions, or researching topics:
        * "What is the latest research on large language models?"
        * "Find information about climate change impacts"
        * "Who invented the internet?"
        * "What are the symptoms of COVID-19?"
        * "Search for recent advancements in renewable energy"
        * "Find tutorials on Python programming"
        * "What happened in world news today?"
        * "Look up the definition of quantum computing"
        
    - Route to "productivity_assistant" when the query involves personal organization, scheduling, reminders, or task management:
        * "Create a to-do list for my project"
        * "Schedule a meeting for tomorrow at 2pm"
        * "Remind me to call John in 3 hours"
        * "Draft an email to my team about the upcoming deadline"
        * "Help me organize my work schedule"
        * "Create a shopping list for dinner ingredients"
        * "Set up a daily reminder for medication"
        * "Plan my study schedule for next week"
        
    - Route to "data_analysis_agent" when the query involves analyzing, visualizing, or processing data:
        * "Analyze this CSV file of sales data" 
        * "Create a chart showing revenue trends"
        * "Calculate the mean and standard deviation of these numbers"
        * "Compare these two datasets and find correlations"
        * "Visualize this data as a histogram"
        * "Perform sentiment analysis on these customer reviews"
        * "Extract insights from this survey data"
        * "Identify patterns in this time series data"
        * "Generate a report based on these quarterly figures"
        
    When in doubt about which agent to route to, consider:
    - If the query is primarily about finding or retrieving information, use "smart_search"
    - If the query is about personal task management or organization, use "productivity_assistant"
    - If the query involves working with or making sense of data, use "data_analysis_agent"
                """
    
                from langchain_core.output_parsers import JsonOutputParser
                from langchain_core.prompts import PromptTemplate

                parser = JsonOutputParser(pydantic_object=Router)
                prompt = PromptTemplate(
                    template="Answer the user query.\n{system_prompt}\n{format_instructions}\nUser: {input}\n",
                    input_variables=["system_prompt","input"],
                    partial_variables={"format_instructions": parser.get_format_instructions()},
                )

                try:
                    router_chain = prompt | llm_router | parser
                    response = router_chain.invoke({"system_prompt":system_prompt, "input": user_input})
                    print("Structured response:", response)
                    return response.get('next', 'smart_search')  # Default to smart_search instead of literature
                except Exception as e:
                    print(f"Structured output parsing error: {e}")
                    # Default to smart_search instead of literature if there's an error
                    return Router(next="smart_search").next
                
                # from langchain_core.output_parsers import JsonOutputParser
                # prompt = ChatPromptTemplate.from_messages([
                #     ("system", system_prompt),
                #     ("user", "{input}")],
                #     partial_variables={"format_instructions": parser.get_format_instructions()})

            #inner_auto
            if st.session_state.global_task in ["Auto Selection", "Inner Auto Selection"]:
                user_input = None  
                # col1, col2 = st.columns([0.75, 13]) - THIS WILL STOP AUTO-SCROLL
                if st.session_state.first_question and st.session_state.first_question[-1]["role"] == "initializer" and len(st.session_state.first_question) % 2 == 1:
                    question = st.session_state.first_question[-1]["content"]
                else:
                    question = None

                # if st.session_state.second_task or :
                try:
                    user_input = None 
                    total_input = st.chat_input(
                            key="inner_auto_selection", 
                            accept_file  = "multiple",
                            file_type = ["pdf", "jpg", "jpeg", "png", "doc", "docx"],
                            placeholder="Enter your query for either Smart Web Search or Productivity Agent or Data Analysis Agent...",
                        )
                    if total_input:
                        if hasattr(total_input, 'text') and total_input.text:
                            user_input = total_input.text
                        else:
                            user_input = "dummy"
                        if hasattr(total_input, 'files') and total_input.files:
                            files_collected = total_input['files']
                            try:
                                chunk_count = process_pdf(files_collected)
                                print(">>>>>>Embedding")
                                print(chunk_count)
                            except:
                                pass
                        else:
                            files_collected = None

                    try:
                        print("new input")
                        print(user_input)
                        if not user_input:
                            user_input = question
                        if user_input:
                            print(f'============ {user_input}')
                            route_ans = supervisor_node(user_input)
                            print(f'=========== {route_ans}')
                            try:
                                task = "Productivity Assistant" if route_ans == "literature" or files_collected else "sql agent"
                            except:
                                task = "Productivity Assistant" if route_ans == "literature" else "sql agent"

                            print(">>>> Inner Auto Selection Final result:", route_ans)
                            st.session_state.inner_auto = True
                    except Exception as e:
                        route_ans = "literature"
                        print(f"Error Inner/Auto Selection: {e}")
                        task = "Productivity Assistant"
                except Exception as e:
                    print(f"Error Inner/Auto Selection: {e}")
                    st.error(f"Error Inner/Auto Selection: {e}")
                    route_ans = "literature"
                    task = "Productivity Assistant"

            if task == "Productivity Assistant":
                # user_input = st.chat_input("Enter your Biomedical Queries for Coherent literature response...")
                stop_button = st.button("▣", key="stop_button", help="Stop the current task")
                try:
                    if pdf_content:
                        pass
                except UnboundLocalError:
                    pdf_content = None
                
                try:
                    if files_collected:
                        pass
                except UnboundLocalError:
                    files_collected = None

                if "vectorstore" not in st.session_state:
                    st.session_state.vectorstore = None
                
                if (not st.session_state.first_question or st.session_state.inner_auto) and ('user_input' in locals() or 'user_input' in globals()):#new
                    print("inner_loop")
                    print(user_input)
                    pass
                elif st.session_state.first_question and st.session_state.first_question[-1]["role"] == "initializer" and len(st.session_state.first_question) % 2 == 1:
                    print("first_question")
                    user_input = st.session_state.first_question[-1]["content"]
                    files_collected = st.session_state.first_question[-1]["file"]
                    if files_collected:
                        chunk_count = process_pdf(files_collected)
                    st.session_state.first_question[-1]["role"] = "processed"
                    if st.session_state.global_task not in ["Auto Selection", "Inner Auto Selection"]:#new
                        placeholder_ip = st.chat_input(
                            key="literature_query", 
                            accept_file  = "multiple",
                            file_type = ["pdf", "jpg", "jpeg", "png", "doc", "docx"],
                            placeholder="Enter your query for general-purpose AI model...",
                        )
                else:
                    print("> not first_question")
                    if st.session_state.global_task not in ["Auto Selection", "Inner Auto Selection"]:#new
                        user_input =  None
                        total_input = st.chat_input(
                            key="literature_query_continue", 
                            accept_file  = "multiple",
                            file_type = ["pdf", "jpg", "jpeg", "png", "doc", "docx"],
                            placeholder="Enter your query for general-purpose AI model...",
                        )
                    
                        if total_input:
                            if hasattr(total_input, 'text') and total_input.text:
                                user_input = total_input.text
                            else:
                                user_input = "dummy"
                            if hasattr(total_input, 'files') and total_input.files:
                                files_collected = total_input['files']
                                try:
                                    chunk_count = process_pdf(files_collected)
                                    print(">>>>>>Embedding")
                                    print(chunk_count)
                                except:
                                    pass
                            else:
                                files_collected = None
                try:
                    if user_input:
                        st.session_state.messages.append({"role": "user", "content": user_input})

                        assistant_avatar = process_avatar("./assets/logo.png")  # Original in assets folder
                        user_avatar = "./assets/logo.png"

                        with st.chat_message("user", avatar=user_avatar):
                            st.markdown(user_input)
                            
                        with st.chat_message("assistant", avatar=assistant_avatar):
                            # First display PDF processing information if available
                            if hasattr(st.session_state, "pdf_processed"):
                                pdf_info = st.session_state.pdf_processed
                                st.write(f"✅ **Successfully processed your PDF into {pdf_info['num_chunks']} knowledge chunks!**")
                                with st.expander("📚 **Chunks Visualization** - gte-Qwen2-1.5B-instruct", expanded=False):
                                    st.write(f":bulb: *{pdf_info['fun_fact']}*")
                                    if pdf_info['num_chunks'] > 0:
                                        st.write(f"✅ **Added your PDF into VectorDB which might contains other documents of the session.**")
                                        st.write("Here's how your document was divided:")
                                        st.bar_chart(pdf_info['chunk_sizes'])
                                    st.markdown("---")
                                    heading_relevant_chunk_placeholder = st.empty()
                                    relevant_chunk_placeholder = st.empty()
                                
                                # Clear the PDF processed info to avoid showing it again
                                del st.session_state.pdf_processed

                            st.session_state["final_output"] = ""
                            st.session_state["is_expanded"] = False
                            st.session_state["full_response"] = ""
                            st.session_state.stop_streaming = False
                            start_time = time.time()
                            # Render expander only during processing
                            # if not st.session_state.processing_done:
                            full_response = ""
                            
                            with st.expander(":bulb: **Hide Reasoning** - Model XYZ",  expanded=True):
                                thinking_placeholder = st.empty()
                            final_placeholder = st.empty()
                            
                            if files_collected and st.session_state.vectorstore:
                                embeddings = model_pdf #HuggingFaceEmbeddings(model_name=str(path_manager.model_paths["EMBEDDINGS_qwen"]))
                                
                                if any(keyword in user_input.lower() for keyword in ["abstract", "conclusion", "introduction"]):
                                    relevant_docs = st.session_state.vectorstore.similarity_search(user_input, k=8,
                                        fetch_k=20,     # Fetch more documents to consider before selecting top 8
                                        lambda_mult=0.7 # Balance between relevance and diversity
                                    )
                                    res = ""
                                    heading_relevant_chunk_placeholder.markdown(f"<h4 style='margin-top:0.7rem' class='biological-title-start'>Best {len(relevant_docs)} Relevant Chunks:</h5>", unsafe_allow_html=True)
                                    for rank, doc in enumerate(relevant_docs, start=1):  
                                        content = doc.page_content  
                                        res += f"**Rank {rank}:**\n\n{content}\n\n---\n\n"
                                    relevant_chunk_placeholder.markdown(res)
                                elif any(keyword in user_input.lower() for keyword in ["summarize", "summarise", "summary", "summarization", "explain", "simplify", "simple term"]):
                                    relevant_docs = st.session_state.vectorstore.similarity_search(user_input, k=20, # Number of documents to return
                                            fetch_k=40,     # Fetch more documents, then select most diverse k
                                            lambda_mult=0.6 # 60% relevance with 40% diversity (lower is more diverse)
                                        )
                                    res = ""
                                    heading_relevant_chunk_placeholder.markdown(f"<h4 style='margin-top:0.7rem' class='biological-title-start'>Best {len(relevant_docs)} Relevant Chunks:</h5>", unsafe_allow_html=True)

                                    for rank, doc in enumerate(relevant_docs, start=1):  
                                        content = doc.page_content  
                                        res += f"**Rank {rank}:**\n\n{content}\n\n---\n\n"
                                    relevant_chunk_placeholder.markdown(res)
                                else:
                                    relevant_docs = st.session_state.vectorstore.similarity_search(user_input, k=5,
                                        fetch_k=15,     # Fetch more documents for better selection
                                        lambda_mult=0.7 # 0.7 balances relevance with diversity (higher = more relevance)
                                    )
                                    res = ""
                                    heading_relevant_chunk_placeholder.markdown(f"<h4 style='margin-top:0.7rem' class='biological-title-start'>Best {len(relevant_docs)} Relevant Chunks:</h5>", unsafe_allow_html=True)
                                    for rank, doc in enumerate(relevant_docs, start=1):  
                                        content = doc.page_content  
                                        res += f"**Rank {rank}:**\n\n{content}\n\n---\n\n"
                                    relevant_chunk_placeholder.markdown(res)


                                pdf_context = "\n\n".join([doc.page_content for doc in relevant_docs])
                                st.session_state.pdf_context = pdf_context
                            else:
                                # pdf_context = ""  # Ensure it's an empty string when no file is uploaded
                                pass
                            

                            # Streaming logic
                            in_thinking = True    
                            raw_question = user_input
                            user_input += f"""Follow all the rules. **Note:** Always use single backticks `keyword` to highlight important keywords.(e.g. `AI Agents`, `LangGraph`, `1999`)
                                **Rules:**
                                - Always include atleast one Table since information can be better presented Factually using **Table** try to include **Table** for compact facts.                       
                                - If required use triple dash like ('---') to seperate different sections for better mark-down layout
                                - Use single or triple backticks `keyword` to highlight important keywords for easy reading and appeal. 
                                - First go through the reasoning steps using appropriate opening and closing tags like <think>...</think> Actual Response.
                                """
                            try:
                                # Get pdf_context from session state instead of assuming it exists as a local variable
                                pdf_context = st.session_state.get('pdf_context', '')
                                if pdf_context:
                                    prompt = get_prompt(user_input, st.session_state.context, pdf_context, pdf=True)   
                                else:
                                    prompt = get_prompt(user_input, st.session_state.context)  
                            except Exception as e:
                                st.error(f"Error processing PDF context: {e}")
                                prompt = get_prompt(user_input, st.session_state.context)

                            gc.collect()

                            torch.cuda.empty_cache()
                            device_id = 0
                            # torch.cuda.set_device(device_id) 

                            if 'full_resoning' not in st.session_state:
                                st.session_state.full_resoning = ""
                            else:
                                st.session_state.full_resoning = ""

                        #     for chunk in llm_literature.stream(prompt):#, max_tokens=8000, top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty, temperature=temperature, stream=True):
                        #         if st.session_state.stop_streaming:
                        #             st.session_state.stop_streaming = True
                        #             break
                                    
                        #         chunk_text = chunk # chunk.content
                        #         st.session_state.full_response += chunk_text

                        #         if in_thinking:
                        #             thinking_placeholder.markdown(st.session_state.full_response + "▌")
                        #             st.session_state.full_resoning += chunk_text

                        #         if "</think>" in st.session_state.full_response.lower():
                        #             in_thinking = False
                        #             parts = st.session_state.full_response.split("</think>", 1)
                        #             st.session_state.final_output = parts[1] if len(parts) > 1 else ""
                        #             # st.session_state.reasoning = parts[0].replace("</think>", "")
                        #             st.session_state.full_response = parts[0].replace("</think>", "")
                        #             st.session_state.is_expanded = False
                        #             # st.session_state.processing_done = True
                        #             time.sleep(0.05)  # Helps sync state changes

                        #         if not in_thinking and chunk_text != "</think>":
                        #             st.session_state.final_output += chunk_text
                        #             final_placeholder.markdown(st.session_state.final_output)

                        #     # Add timing stats
                        #     execution_time = time.time() - start_time
                        #     ntokens = len(st.session_state.full_response.split())
                        #     timing_stats = f'<span style="color: grey; font-size: 0.8em; font-style: italic;">Literature ⏳ {execution_time:.2f}s | {(execution_time/ntokens):.3f}s/token | {round((ntokens/execution_time),1)} tokens/s</span>'
                        #     final_placeholder.markdown(f"{st.session_state.final_output}<br>{timing_stats}", unsafe_allow_html=True)
                            
                        # # Update conversation history
                        # st.session_state.messages.append({"role": "assistant", "reasoning": st.session_state.full_resoning, "content": st.session_state.final_output})
                        

                        st.markdown("""
                        <script>
                        /* Reposition the file upload button container */
                        [data-testid="stChatInputFileUploadButton"] {
                            margin-right: 10px !important;
                            border: none !important;
                            position: absolute !important;
                            bottom: -17px !important;
                            z-index: 10 !important;
                            order: 2 !important;
                        }

                        /* Reposition line next 2 file upload button */ 
                        .st-emotion-cache-g39xc6  {
                            margin-left: 29px !important;
                            margin-right: 0px !important;
                            border: none !important;
                            position: absolute !important;
                            bottom: -20px !important;
                            background-color: #e5e4d5 !important;
                            z-index: 10 !important;
                            order: 2 !important;
                            min-height: 40px !important;
                        }</script>""", unsafe_allow_html=True) 
                except Exception as e:
                    print("Except literature block:")
                    print(st.session_state.global_task)
                    import traceback
                    def get_clean_error_message(e: Exception) -> str:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        tb_last = "".join(traceback.format_exception_only(type(e), e))
                        return f"{error_type}: {error_msg}\n{tb_last}"
                    st.markdown(
                        f"Error details: {get_clean_error_message(e)}\n"
                    )

            # productivity agent
            else:
                if (not st.session_state.first_question or st.session_state.inner_auto) and ('user_input' in locals() or 'user_input' in globals()):#new
                    print("<<<<<<<<<< inner_loop_productivity_db")
                    print(user_input)
                    question = user_input
                    pass
                elif st.session_state.first_question and st.session_state.first_question[-1]["role"] == "initializer" and len(st.session_state.first_question) % 2 == 1:
                    user_input = None #new
                    question = st.session_state.first_question[-1]["content"]
                    st.session_state.first_question[-1]["role"] = "processed"
                    if st.session_state.global_task not in ["Auto Selection", "Inner Auto Selection"]:#new
                        placeholder_ip = st.chat_input(f"Enter your {db_name} data related Query...")
                else:
                    user_input = None #new
                    if st.session_state.global_task not in ["Auto Selection", "Inner Auto Selection"]:#new
                        question = st.chat_input(f"Enter your {db_name} data related Query...")
                stop_button = st.button("▣", key="stop_button", help="Stop the current task")
                # # Add user message to chat history
                if question:
                    st.session_state.messages.append({"role": "user", "content": question})
                    assistant_avatar = process_avatar("./assets/logo.png")  # Original in assets folder
                    user_avatar = "./assets/logo.png"
                    
                    with st.chat_message("user", avatar=user_avatar):
                        st.markdown(question)
                    # with st.chat_message("assistant", avatar=assistant_avatar):
                       
            TEMP, TEMP2 = 0, 0    
            if st.session_state.global_task in ["Auto Selection", "Inner Auto Selection"]:
                print("if block agent")
                st.session_state.second_task = True
                print(st.session_state.global_task)
                print(st.session_state.second_task)
                # st.rerun()  
            else:
                print(st.session_state.global_task)
                print('else block agent')  
    except Exception as e:
        st.markdown(f'{str(e)}')
        
if __name__ == "__main__":    
    main()
