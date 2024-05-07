from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
import glob
from PyPDF2 import PdfReader
from tqdm import tqdm
import pandas as pd

api_key = os.environ.get('OPENAI_API_KEY')


if __name__ == "__main__":
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    folder_path = '/Users/rashisolanki/Desktop/CAPSTONE/RAG_ChatBot/documents'

    raw_text = ''

    for filename in tqdm(os.listdir(folder_path)):
        print(f"Processing file: {filename}")
        # Check if the file is a PDF
        if filename.endswith('.pdf'):
            pdf_file_path = os.path.join(folder_path, filename)
            reader = PdfReader(pdf_file_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text
        # Check if the file is an XLSX
        elif filename.endswith('.xlsx'):
            excel_file_path = os.path.join(folder_path, filename)
            reader = pd.read_excel(excel_file_path, sheet_name=None)
            for sheet_name, df in reader.items():
                text = df.to_string(index=False)
                raw_text += f"\n\n{sheet_name}\n\n{text}"

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1300,
        chunk_overlap  = 200,
        length_function = len,
        )
    texts = text_splitter.split_text(raw_text)
    
    embeddings = OpenAIEmbeddings(api_key=api_key)
    docsearch = FAISS.from_texts(texts, embeddings)
    docsearch.save_local(DB_FAISS_PATH)
    