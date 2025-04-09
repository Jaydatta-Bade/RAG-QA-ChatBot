import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader


def load_document(file):
    name, extension = os.path.splitext(file)


    if extension == '.pdf':
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        loader = Docx2txtLoader(file)
    elif extension == '.txt':  
        loader = TextLoader(file)
    else:
        raise ValueError('Document format is not supported!')


    data = loader.load()
    return data
