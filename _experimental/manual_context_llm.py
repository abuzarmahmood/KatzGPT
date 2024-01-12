"""
https://levelup.gitconnected.com/langchain-for-multiple-pdf-files-87c966e0c032
https://medium.com/data-professor/beginners-guide-to-openai-api-a0420bc58ee5
"""

from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from glob import glob
import os
from tqdm import tqdm
from joblib import Parallel, delayed 
from pickle import dump, load
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

##############################
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
# llm = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo-0613")
llm = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo-1106")
# llm = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo-instruct")

# llm = OpenAI(temperature=0.8)
question_generator = LLMChain(
        llm=llm,
        prompt=QA_PROMPT,
        verbose=True,
    )

doc_chain = load_qa_with_sources_chain(llm,
                                       chain_type = 'stuff')

memory = ConversationBufferMemory(
        memory_key="chat_history", 
        input_key="question",
        output_key="answer",
        return_messages=True)
retriever = vectordb.as_retriever(
        search_kwargs = {"k":50})
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
query = "Protocol for conditioned taste aversion" + \
        " Give detailed answer" 
result = pdf_qa({"question": query})
for this_key in result.keys():
    if this_key not in ['chat_history', 'source_documents']:
        print()
        print(f"{this_key} : {result[this_key]}")

############################################################
# Direct search
# query = 'global workspace' 
# docs = vectordb.similarity_search(query, k = 50)
outs = vectordb.similarity_search_with_relevance_scores(query, k = 10)
docs, scores = zip(*outs)

docs_sources = list(set([doc.metadata['source'] for doc in docs]))

for this_source, this_score in zip(docs_sources, scores): 
    print(this_source.split('/')[-1])
    print(this_score)
    print('===========================')

{"\n".join([doc.page_content for doc in docs])}
############################################################
query = "Papers by Donald Katz" 

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

print(completion.choices[0].message.content)
