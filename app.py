"""
This is the main app file for the Streamlit app.
"""

############################################################
## Imports 
############################################################

import streamlit as st
from dataclasses import dataclass
from rag_llm import run_query, return_vectordb

############################################################
# Get Vector Database 
############################################################
vectordb = return_vectordb()

############################################################
# Run Chat
############################################################


@dataclass
class Message:
    actor: str
    payload: str


USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"
if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = [Message(
        actor=ASSISTANT,
        payload="Welcome to KatzGPT! I'm an AI assistant for answering questions about systems neuroscience, specifically taste processing. Ask me a question about taste processing and I'll try to answer it.")]

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("Enter a prompt here")

if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)
    # response: str = f"You wrote {prompt}"
    response: str = run_query(prompt, vectordb, k=5)
    st.session_state[MESSAGES].append(
        Message(actor=ASSISTANT, payload=response))
    st.chat_message(ASSISTANT).write(response)
