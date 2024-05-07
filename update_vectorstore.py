#import Essential dependencies

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from PyPDF2 import PdfReader
import pandas as pd

api_key = os.environ.get('OPENAI_API_KEY')

# to add more docs to the vectorstore
def add_to_vectore_store(folder_path, db):
        for filename in os.listdir(folder_path):
                if filename.endswith('.pdf'):
                        pdf_file_path = os.path.join(folder_path, filename)
                        loader = PdfReader(pdf_file_path)
                elif filename.endswith('.xlsx'):
                        excel_file_path = os.path.join(folder_path, filename)
                        loader = pd.read_excel(excel_file_path, sheet_name=None)
                else:
                        continue  # Skip files that are neither PDFs nor XLSXs

                raw_text = loader.load()
                text_splitter = CharacterTextSplitter(        
                                separator = "\n",
                                chunk_size = 1300,
                                chunk_overlap  = 200,
                                length_function = len,
                                )
                texts = text_splitter.split_text(raw_text)
                
                embeddings = OpenAIEmbeddings(api_key=api_key)
                temp_db = FAISS.from_texts(texts, embeddings)
                db.merge_from(temp_db)

if __name__=="__main__":
    embeddings=OpenAIEmbeddings(api_key=api_key)
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)  
    folder_path = '...'
    add_to_vectore_store(folder_path, db)
  
