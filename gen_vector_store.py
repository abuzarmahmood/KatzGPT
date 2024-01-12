"""
Generate a vector store for use in RAG-LLM given
PDFs of scientific articles.
"""

############################################################
## Imports 
############################################################

from glob import glob
import os
from tqdm import tqdm
from joblib import Parallel, delayed 
from pickle import dump, load
from utils import return_paths

from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 

############################################################
## Convenience Functions 
############################################################

def parallelize(data, func, num_of_processes=8):
    """
    Parallelize a function across a list of data.

    Inputs:
        data: list of data to be processed
        func: function to be applied to each element of data
        num_of_processes: number of processes to use in parallelization

    Returns:
        results: list of results from applying func to data
    """
    return Parallel(n_jobs=num_of_processes)(delayed(func)(i) for i in tqdm(data))

def try_load(this_path):
    """
    Try to load a document from a path.

    Inputs:
        this_path: path to a document

    Returns:
        docs: list of documents
    """
    try:
        loader = PyPDFLoader(this_path)
        docs = loader.load()
        return docs
    except:
        print(f'Load failed : {this_path}')
        return None

############################################################
## Generate Docs
############################################################

(
    file_list, 
    docs_output_path, 
    docs_output_dir,
    vector_persist_dir,
    ) = return_paths()


if not os.path.exists(docs_output_path):
    docs_list = parallelize(file_list, try_load, num_of_processes=24)
    # Drop None
    docs_list = [doc for doc in docs_list if doc is not None]
    # Flatten list
    docs_list = [item for sublist in docs_list for item in sublist]
    # Extract document source from each document
    doc_source = [doc.metadata['source'] for doc in docs_list]
    ## Count length of set of document source
    #len(set(doc_source))
    # Save docs
    with open(docs_output_path, 'wb') as f:
        dump(docs_list, f)
else:
    docs_list = load(open(docs_output_path, 'rb')) 

############################################################
# Generate Embeddings
############################################################

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents([docs_list[0]], embedding=embeddings, 
                              persist_directory=vector_persist_dir)
vectordb.persist()
for doc in tqdm(docs_list):
    vectordb.add_documents([doc])
