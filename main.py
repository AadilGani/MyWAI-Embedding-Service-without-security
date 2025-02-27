from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import requests
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import io
import random
import string
import json
import hashlib
import uuid
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table, select, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB, UUID
from pgvector.sqlalchemy import Vector
from langchain_community.embeddings import OllamaEmbeddings

app = FastAPI()

embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://sestrilevante.platform.myw.ai:11434')

CONNECTION_STRING = "postgresql+psycopg2://mywai:Bth-12345@localhost:5432/experiment"

def process_pdf(file):
    with open("temp_pdf.pdf", "wb") as f:
        f.write(file)
    loader = PyPDFLoader("temp_pdf.pdf")
    pdf_text = loader.load()
    return pdf_text

def process_docx(file):
    with open("temp_docx.docx", "wb") as f:
        f.write(file)
    loader = UnstructuredWordDocumentLoader("temp_docx.docx")
    docx_text = loader.load()
    return docx_text

def process_txt(file):
    with open("temp_txt.txt", "wb") as f:
        f.write(file)
    loader = TextLoader("temp_txt.txt")
    txt_text = loader.load()
    return txt_text

def process_csv(file):
    with open("temp_csv.csv", "wb") as f:
        f.write(file)
    loader = CSVLoader("temp_csv.csv")
    csv_text = loader.load()
    return csv_text

def process_xlsx(file):
    with open("temp_xlsx.xlsx", "wb") as f:
        f.write(file)
    loader = UnstructuredExcelLoader("temp_xlsx.xlsx")
    xlsx_text = loader.load()
    return xlsx_text

def split_text(docs, chunk_size=3000, overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in docs:
        chunks.extend(text_splitter.split_text(doc.page_content))
    return chunks

def create_embeddings(text_chunks, embeddings):
    return [embeddings.embed_query(chunk) for chunk in text_chunks]

Base = declarative_base()

class LangchainPgEmbedding(Base):
    __tablename__ = 'langchain_pg_embedding'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pageContent = Column(String, name='pageContent')
    file_metadata = Column(JSONB, name='metadata')
    embedding = Column(Vector, name='embedding')
    equipment_id = Column(Integer, name='equipment_id')
    document_id = Column(Integer, name='document_id')

# Connect to PostgreSQL
engine = create_engine(CONNECTION_STRING)
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

def alter_table_add_columns():
    with engine.connect() as connection:
        metadata = MetaData()
        text_embeddings_table = Table('langchain_pg_embedding', metadata, autoload_with=engine)
        if not any(col.name == 'document_id' for col in text_embeddings_table.columns):
            connection.execute(text('ALTER TABLE langchain_pg_embedding ADD COLUMN document_id INTEGER'))
        if not any(col.name == 'equipment_id' for col in text_embeddings_table.columns):
            connection.execute(text('ALTER TABLE langchain_pg_embedding ADD COLUMN equipment_id INTEGER'))
        if not any(col.name == 'metadata' for col in text_embeddings_table.columns):
            connection.execute(text('ALTER TABLE langchain_pg_embedding ADD COLUMN metadata JSONB'))

def build_metadata(pdf_docs, file_name):
    first_doc = pdf_docs[0].metadata
    return {
        "loc": {"pageNumber": 1},
        "pdf": {
            "info": first_doc.get('info', {}),
            "version": first_doc.get('version', ''),
            "metadata": first_doc.get('metadata', {}),
            "totalPages": len(pdf_docs)
        },
        "source": "blob",
        "blobType": file_name.split('.')[-1].lower()
    }

def insert_embeddings(text_chunks, embeddings_list, document_id, equipment_id, metadata):
    for content, embedding in zip(text_chunks, embeddings_list):
        sanitized_content = content.replace('\x00', '')
        chunk_id = uuid.uuid4()
        exists = session.query(LangchainPgEmbedding).filter_by(pageContent=sanitized_content).first()
        if exists:
            continue
        new_entry = LangchainPgEmbedding(
            id=chunk_id,
            pageContent=sanitized_content,
            file_metadata=metadata,
            embedding=np.array(embedding).flatten().tolist(),
            equipment_id=equipment_id,
            document_id=document_id
        )
        session.add(new_entry)
    session.commit()

class GenerateEmbeddingsResponse(BaseModel):
    message: str

@app.post("/generate_embeddings", response_model=GenerateEmbeddingsResponse)
async def generate_embeddings(file: UploadFile = File(...)):
    print("")
    if not file:
        raise HTTPException(status_code=400, detail="File is required")

    alter_table_add_columns()

    file_extension = file.filename.split('.')[-1].lower()
    file_content = await file.read()

    try:
        if file_extension == 'pdf':
            file_processed = process_pdf(file_content)
        elif file_extension == 'docx':
            file_processed = process_docx(file_content)
        elif file_extension == 'txt':
            file_processed = process_txt(file_content)
        elif file_extension == 'csv':
            file_processed = process_csv(file_content)
        elif file_extension == 'xlsx':
            file_processed = process_xlsx(file_content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        text_chunks = split_text(file_processed)
        embeddings_list = create_embeddings(text_chunks, embeddings)

        document_id = random.randint(1, 10000)
        equipment_id = random.randint(1, 10000)

        metadata = build_metadata(file_processed, file.filename)

        insert_embeddings(text_chunks, embeddings_list, document_id, equipment_id, metadata)
        return GenerateEmbeddingsResponse(message="Embeddings created and stored successfully!")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=1211)