import os
import torch
import ollama
import yaml
import faiss
import numpy as np
from PyPDF2 import PdfReader
import uuid
import pickle
import re
import string

PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def load_config(config_file):
    print(PINK + "Loading configuration..." + RESET_COLOR)
    try:
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Configuration file '{config_file}' not found.")
        exit(1)

def open_file(filepath):
    print(PINK + "Opening file..." + RESET_COLOR)
    try:
        if filepath.endswith('.pdf'):
            reader = PdfReader(filepath)
            text = ''
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += clean_text_for_embedding(page_text) + '\n'
            return text
        else:
            with open(filepath, 'r', encoding='utf-8') as infile:
                text = infile.read()
                return clean_text_for_embedding(text)
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error opening file '{filepath}': {str(e)}")
        return None

def initialize_faiss(index_file, documents_file, embedding_dim=1024):
    print(PINK + "Initializing Faiss..." + RESET_COLOR)
    if os.path.exists(index_file):
        with open(index_file, 'rb') as f:
            index, documents = pickle.load(f)
        if index.d != embedding_dim:
            print(f"Embedding dimension mismatch: index dimension is {index.d}, expected {embedding_dim}")
            exit(1)
    else:
        index = faiss.IndexFlatL2(embedding_dim)
        documents = {}
    return index, documents

def save_faiss_index(index_file, documents_file, index, documents):
    with open(index_file, 'wb') as f:
        pickle.dump((index, documents), f)
    with open(documents_file, 'wb') as f:
        pickle.dump(documents, f)

def load_or_generate_embeddings(index, documents, vault_content, embedding_model, index_file, documents_file):
    print(PINK + "Loading or generating embeddings..." + RESET_COLOR)
    for content in vault_content:
        try:
            response = ollama.embeddings(model=embedding_model, prompt=content)
            embedding = np.array(response["embedding"]).astype('float32')
            print(f"Generated embedding with shape: {embedding.shape}")  # Print the shape of the embedding
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            continue

        doc_id = str(uuid.uuid4())
        index.add(np.expand_dims(embedding, axis=0))
        documents[doc_id] = content

    save_faiss_index(index_file, documents_file, index, documents)
    return index, documents

def get_relevant_context(query, index, documents, top_k, embedding_model):
    print(PINK + "Retrieving relevant context..." + RESET_COLOR)
    try:
        response = ollama.embeddings(model=embedding_model, prompt=query)
        query_embedding = np.array(response["embedding"]).astype('float32')
        print(f"Query embedding with shape: {query_embedding.shape}")  # Print the shape of the query embedding
        D, I = index.search(np.expand_dims(query_embedding, axis=0), top_k)
        return [documents[list(documents.keys())[i]] for i in I[0]]
    except Exception as e:
        print(f"Error getting relevant context: {str(e)}")
        return []

def ollama_chat(user_input, system_message, index, documents, qa_model, embedding_model, conversation_history, top_k, client, document_content=None):
    if document_content:
        context_str = document_content
        #print("Using selected document content: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        relevant_context = get_relevant_context(user_input, index, documents, top_k, embedding_model)
        if relevant_context:
            context_str = "\n".join(relevant_context)
            print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
        else:
            print("No relevant context found.")
            context_str = ""

    user_input_with_context = user_input
    if context_str:
        user_input_with_context = context_str + "\n\n" + user_input

    conversation_history.append({"role": "user", "content": user_input_with_context})
    messages = [{"role": "system", "content": system_message}, *conversation_history]

    try:
        response = client.chat.completions.create(
            model=qa_model,
            messages=messages
        )
        conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in Ollama chat: {str(e)}")
        return "An error occurred while processing your request."

def list_documents(documents):
    print(PINK + "Listing documents..." + RESET_COLOR)
    return list(documents.keys())

def get_document_by_id(documents, doc_id):
    print(PINK + f"Retrieving document with ID {doc_id}..." + RESET_COLOR)
    return documents.get(doc_id, None)

def clean_text_for_embedding(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

def clean_text_for_qa(text):
    text = ' '.join(text.split())
    return text
