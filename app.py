import streamlit as st
from dataclasses import dataclass

############################################################
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from glob import glob
import os
from tqdm import tqdm
from joblib import Parallel, delayed 
from pickle import dump, load
import numpy as np
from pprint import pprint

from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.prompt import PromptTemplate
import openai

############################################################
## Generate Docs
############################################################
docs_path = '/media/bigdata/projects/istyar/data/abu_zotero'
file_list = glob(os.path.join(docs_path, "*"))
vector_persist_dir = '/media/bigdata/projects/katzGPT/vector_store'
docs_output_dir = '/media/bigdata/projects/katzGPT/docs'
docs_output_path = os.path.join(docs_output_dir, 'docs.pkl')

if not os.path.exists(vector_persist_dir):
    os.makedirs(vector_persist_dir)

if not os.path.exists(docs_output_dir):
    os.makedirs(docs_output_dir)

docs_list = load(open(docs_output_path, 'rb')) 

############################################################
# Generate Embeddings
############################################################
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=vector_persist_dir, 
                  embedding_function=embeddings,
                  collection_metadata={"hnsw:space": "cosine"})

template = """You are an AI assistant for answering questions about systems neuroscience, specifically taste processing.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""

QA_PROMPT = PromptTemplate(
        template=template, 
        input_variables=[
                       "question", 
                       "context"
                       ]
        )

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo-0613")
# llm = ChatOpenAI(temperature=0.8, model="gpt-4-0613")

# llm = OpenAI(temperature=0.8)
question_generator = LLMChain(
        llm=llm,
        prompt=QA_PROMPT,
        verbose=True,
    )

doc_chain = load_qa_with_sources_chain(llm,
                                       chain_type = 'stuff')

def get_docs(query):
    outs = vectordb.similarity_search_with_relevance_scores(query, k = 5)
    docs, scores = zip(*outs)

    docs_sources = list(set([doc.metadata['source'] for doc in docs]))

    docs_str = ""
    docs_str += '\n'
    docs_str += "=== Sources ===\n\n"
    for this_source, this_score in zip(docs_sources, scores): 
        docs_str += f"{this_source.split('/')[-1]}"
        docs_str += f" === Score: {np.round(this_score,2)}\n\n"
    return docs_str

def run_query2(
        query,
        question_generator,
        doc_chain,
        ):
    outs = vectordb.similarity_search_with_relevance_scores(query, k = 10)
    docs, scores = zip(*outs)

    prompt = f"""
    Question: {query}
    =========
    {docs}
    =========
    Answer in Markdown:"""

    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-1106",
      messages=[
        {"role": "system", "content": "You are an AI assistant for answering questions about systems neuroscience, specifically taste processing. You are given the following extracted parts of a long document and a question. Provide a conversational answer. Always indicate your sources"}, 
        {"role": "user", "content": prompt}
      ],
    )

    out_str = completion.choices[0].message.content
    docs_str = get_docs(query)
    out_str += '\n\n'
    out_str += docs_str
    return out_str


def run_query(
        query,
        question_generator,
        doc_chain,
        ):
    memory = ConversationBufferMemory(
            memory_key="chat_history", 
            input_key="question",
            output_key="answer",
            return_messages=True)
    retriever = vectordb.as_retriever(
            search_kwargs = {"k":5})
    pdf_qa = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
        memory=memory,
        max_tokens_limit = 1000,
        rephrase_question = False,
        verbose=True,
    )
    # query = "Protocol for conditioned taste aversion" + \
    #         " Give detailed answer" 
    result = pdf_qa({"question": query})
    out_str = ""
    for this_key in result.keys():
        if this_key not in ['chat_history', 'source_documents']:
            #print()
            #print(f"{this_key} : {result[this_key]}")
            out_str += '\n'
            out_str += f"{this_key} : {result[this_key]}\n"

    docs_str = get_docs(query)
    out_str += '\n\n'
    out_str += docs_str
    return out_str

run_query_simple = lambda query: run_query2(query,
                                           question_generator=question_generator,
                                           doc_chain=doc_chain,
                                           )


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
    response: str = run_query_simple(prompt)
    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
    st.chat_message(ASSISTANT).write(response) 
    # Write each line of the response separately
    # unless it is a blank line
    # for line in response.splitlines():
    #     if line:
    #         st.chat_message(ASSISTANT).write(line)
