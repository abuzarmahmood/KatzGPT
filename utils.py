"""
This file contains the paths for the project and some helper functions.
"""

import os
from glob import glob


def return_paths():
    """
    Return paths for the project.
    """
    docs_path = '/media/bigdata/projects/istyar/data/abu_zotero'
    file_list = glob(os.path.join(docs_path, "*"))
    vector_persist_dir = '/media/bigdata/projects/katzGPT/vector_store'
    docs_output_dir = '/media/bigdata/projects/katzGPT/docs'
    docs_output_path = os.path.join(docs_output_dir, 'docs.pkl')
    return file_list, docs_output_path, docs_output_dir, vector_persist_dir


# Run this to generate the paths
(
    file_list, 
    docs_output_path, 
    docs_output_dir,
    vector_persist_dir,
    ) = return_paths()

if not os.path.exists(vector_persist_dir):
    os.makedirs(vector_persist_dir)

if not os.path.exists(docs_output_dir):
    os.makedirs(docs_output_dir)
