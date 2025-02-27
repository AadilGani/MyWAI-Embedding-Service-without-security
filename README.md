# MyWAI-Embedding-Service-without-security
This repository contains a FastAPI-based embedding service for MyWAI, designed to process and store embeddings from various file types (PDF, DOCX, TXT, CSV, XLSX) using OllamaEmbeddings. The service reads files, splits the text into chunks, generates embeddings, and stores them in a PostgreSQL database with vector support.

## Note: This service is currently without security measures. It is intended for development and testing purposes only. Do not use this service in a production environment without implementing appropriate security features.

# Features
## Supports multiple file types: PDF, DOCX, TXT, CSV, XLSX.
## Uses OllamaEmbeddings for generating text embeddings.
## Stores embeddings in a PostgreSQL database with vector support.
## Automatically handles table alterations for new columns.
## Generates and stores metadata for each processed file.


# Technologies Used
FastAPI
SQLAlchemy
PostgreSQL
LangChain
OllamaEmbeddings

# Setup
Prerequisites
Python 3.8+
PostgreSQL
pgvector extension for PostgreSQL

# Installation
Clone the repository:

https://github.com/AadilGani/MyWAI-Embedding-Service-without-security.git

# Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install the required dependencies:
pip install -r requirements.txt

# Set up the PostgreSQL database:

Create a new database and user:

CREATE DATABASE experiment;
CREATE USER mywai WITH ENCRYPTED PASSWORD 'Bth-12345';
GRANT ALL PRIVILEGES ON DATABASE experiment TO mywai;

# Update the CONNECTION_STRING in the code with your database credentials if necessary.

# Start the FastAPI server:
uvicorn app:app --host 0.0.0.0 --port 1211

# API Endpoints
Generate Embeddings
URL: /generate_embeddings
Method: POST
Description: Upload a file to generate and store embeddings.
Request:
File: A file in one of the supported formats (PDF, DOCX, TXT, CSV, XLSX).
Response:
200 OK: {"message": "Embeddings created and stored successfully!"}
400 Bad Request: {"detail": "File is required"} or {"detail": "Unsupported file type"}
500 Internal Server Error: {"detail": "Error processing file: <error_message>"}
Contributing
Contributions are welcome! Please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
