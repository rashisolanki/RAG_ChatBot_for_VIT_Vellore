#import Essential dependencies
import streamlit as sl
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
import os.path
from libs.helper import *
import uuid
import pandas as pd
import openai
from requests.models import ChunkedEncodingError
from streamlit.components import v1
import random
import time

api_key = os.environ.get('OPENAI_API_KEY')

sl.set_page_config(page_title='RAGbot for VIT', layout="wide", page_icon=':🤖:')
sl.markdown(css_code, unsafe_allow_html=True)

if "initial_settings" not in sl.session_state:
    sl.session_state["path"] = "history_chats_file"
    sl.session_state["history_chats"] = get_history_chats(sl.session_state["path"])
    sl.session_state["delete_dict"] = {}
    sl.session_state["delete_count"] = 0
    sl.session_state["error_info"] = ""
    sl.session_state["current_chat_index"] = 0
    sl.session_state["user_input_content"] = ""
    sl.session_state["initial_settings"] = True

with sl.sidebar:
    sl.markdown("</br><h1 style='text-align: center;'>🤖 </br> Discover VIT like </br>never before</h1>", unsafe_allow_html=True)
    sl.write("\n")
    chat_container = sl.container(height=65)
    with chat_container:
        current_chat = sl.radio(
            label="Previous History Window",
            format_func=lambda x: x.split("_")[0] if "_" in x else x,
            options=sl.session_state["history_chats"],
            label_visibility="collapsed",
            index=sl.session_state["current_chat_index"],
            key="current_chat"
            + sl.session_state["history_chats"][sl.session_state["current_chat_index"]],
        )
    
    sl.write("\n")

def delete_chat():
    if len(sl.session_state["history_chats"]) == 1:
        chat_init = "New Chat_" + str(uuid.uuid4())
        sl.session_state["history_chats"].append(chat_init)
    pre_chat_index = sl.session_state["history_chats"].index(current_chat)
    if pre_chat_index > 0:
        sl.session_state["current_chat_index"] = (
            sl.session_state["history_chats"].index(current_chat) - 1
        )
    else:
        sl.session_state["current_chat_index"] = 0
    sl.session_state["history_chats"].remove(current_chat)
    remove_data(sl.session_state["path"], current_chat)

    # Clear chat messages when switching chats
    sl.session_state.messages = []

# Reset Chat Option
with sl.sidebar:
    c1 = sl.columns(1)
    delete_chat_button = c1[0].button(
        "Reset Chat", use_container_width=True, key="delete_chat_button"
    )
    if delete_chat_button:
        delete_chat()
        sl.experimental_rerun()

def write_data(new_chat_name=current_chat):
    if "apikey" in sl.secrets:
        sl.session_state["paras"] = {
            "temperature": sl.session_state["temperature" + current_chat],
            "top_p": sl.session_state["top_p" + current_chat],
            "presence_penalty": sl.session_state["presence_penalty" + current_chat],
            "frequency_penalty": sl.session_state["frequency_penalty" + current_chat],
        }
        sl.session_state["contexts"] = {
            "context_select": sl.session_state["context_select" + current_chat],
            "context_input": sl.session_state["context_input" + current_chat],
            "context_level": sl.session_state["context_level" + current_chat],
        }
        save_data(
            sl.session_state["path"],
            new_chat_name,
            sl.session_state["history" + current_chat],
            sl.session_state["paras"],
            sl.session_state["contexts"],
        )

def reset_chat_name(chat_name):
    chat_name = chat_name + "_" + str(uuid.uuid4())
    new_name = filename_correction(chat_name)
    current_chat_index = sl.session_state["history_chats"].index(current_chat)
    sl.session_state["history_chats"][current_chat_index] = new_name
    sl.session_state["current_chat_index"] = current_chat_index
    # write new file
    # write_data(new_name)
    # # transfer data
    sl.session_state["history" + new_name] = sl.session_state["history" + current_chat]
    for item in [
        "context_select",
        "context_input",
        "context_level",
        *initial_content_all["paras"],
    ]:
        sl.session_state[item + new_name + "value"] = sl.session_state[
            item + current_chat + "value"
        ]
    remove_data(sl.session_state["path"], current_chat)

# Rename Chat Option
with sl.sidebar:
    if ("set_chat_name" in sl.session_state) and sl.session_state[
        "set_chat_name"
    ] != "":
        reset_chat_name(sl.session_state["set_chat_name"])
        sl.session_state["set_chat_name"] = ""
        sl.experimental_rerun()

    sl.write("\n")
    sl.write("\n")
    sl.text_input("Rename Chat", key="set_chat_name", placeholder="Click Enter to Rename Chat")
    sl.write("\n")
    sl.write("\n")
    sl.write("\n")

#logo
with sl.sidebar:
     sl.write("\n")
     sl.write("\n")
     sl.write("\n")
     sl.image('vit_logo.png', caption='Made by Rashi Solanki and Veer Sanghavi')

if "history" + current_chat not in sl.session_state:
    for key, value in load_data(sl.session_state["path"], current_chat).items():
        if key == "history":
            sl.session_state[key + current_chat] = value
        else:
            for k, v in value.items():
                sl.session_state[k + current_chat + "value"] = v


#function to rename chat name as the first message
def input_callback():
    if sl.session_state["user_input_area"] != "":
        user_input_content = sl.session_state["user_input_area"]
        df_history = pd.DataFrame(sl.session_state["history" + current_chat])
        if df_history.empty or len(df_history.query('role!="system"')) == 0:
            new_name = extract_chars(user_input_content, 18)
            reset_chat_name(new_name)

# Call input_callback function only once after the first input
if "input_callback_called" not in sl.session_state:
    sl.session_state["input_callback_called"] = True
    query = sl.chat_input("💬 Enter your query", key="user_input_area", on_submit=input_callback)
else:
    query = sl.chat_input("💬 Enter your query", key="user_input_area")

# ----------------------------------------------------------
# RAG MAIN LOGIC
# ----------------------------------------------------------

sl.header("Welcome to the RAGbot for VIT Vellore")
sl.write("🤖 You can chat by Entering your queries ")

#function to load the vectordatabase
def load_knowledgeBase():
        embeddings=OpenAIEmbeddings(api_key=api_key)
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
        
#function to load the OPENAI LLM
def load_llm():
        from langchain_openai import ChatOpenAI
        from langchain import OpenAI #for 3.5 turbo instruct model
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0, api_key=api_key)
        return llm

#creating prompt template using langchain
def load_prompt():
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return qa_prompt

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def response_generator(response):
       for word in response.split():
                yield word + " "
                time.sleep(0.05)

if __name__=='__main__':
        knowledgeBase=load_knowledgeBase()
        llm=load_llm()
        prompt=load_prompt()

        if "messages" not in sl.session_state:
                sl.session_state.messages = []
        for message in sl.session_state.messages:
                with sl.chat_message(message["role"]):
                        sl.markdown(message["content"])

        chat_history = []

        if "user_prompt_history" not in sl.session_state:
            sl.session_state["user_prompt_history"]=[]
        if "chat_answers_history" not in sl.session_state:
            sl.session_state["chat_answers_history"]=[]
        if "chat_history" not in sl.session_state:
            sl.session_state["chat_history"]=[]

        if query:

            with sl.chat_message("user"):
                sl.markdown(query)
            sl.session_state.messages.append({"role": "user", "content": query})

            if "messages" not in sl.session_state:
                sl.session_state.messages = []
            
            qa = ConversationalRetrievalChain.from_llm(
                llm=llm, 
                retriever= knowledgeBase.as_retriever()
                )

            output = qa({"question":query, "chat_history": sl.session_state["chat_history"]})

            sl.session_state["chat_answers_history"].append(output['answer'])
            sl.session_state["user_prompt_history"].append(query)
            sl.session_state["chat_history"].append((query, output['answer']))

            # Display assistant response in chat message container
            with sl.chat_message("assistant"):
                response = sl.write_stream(response_generator(output['answer']))

            # Add assistant response to chat history
            sl.session_state.messages.append({"role": "assistant", "content": response})
