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

DEBUG = False

def set_debug(debug):
    global DEBUG
    DEBUG = debug


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

def initialize_faiss(index_file, documents_file, metadata_file, embedding_dim=1024):
    print(PINK + "Initializing Faiss..." + RESET_COLOR)
    if os.path.exists(index_file) and os.path.exists(metadata_file):
        with open(index_file, 'rb') as f:
            index, documents = pickle.load(f)
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        if index.d != embedding_dim:
            print(f"Embedding dimension mismatch: index dimension is {index.d}, expected {embedding_dim}")
            exit(1)
    else:
        index = faiss.IndexFlatL2(embedding_dim)
        documents = {}
        metadata = {}
    return index, documents, metadata


def save_faiss_index(index_file, documents_file, metadata_file, index, documents, metadata):
    with open(index_file, 'wb') as f:
        pickle.dump((index, documents), f)
    with open(documents_file, 'wb') as f:
        pickle.dump(documents, f)
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

def load_or_generate_embeddings(index, documents, metadata, file_name, vault_content, embedding_model, index_file, documents_file, metadata_file):
    print(PINK + "Loading and generating embeddings..." + RESET_COLOR)
    doc_ids = []
    for content in vault_content:
        try:
            response = ollama.embeddings(model=embedding_model, prompt=content)
            embedding = np.array(response["embedding"]).astype('float32')
            if DEBUG:
                print(f"Generated embedding with shape: {embedding.shape}")
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            continue

        doc_id = str(uuid.uuid4())
        index.add(np.expand_dims(embedding, axis=0))
        documents[doc_id] = content
        doc_ids.append(doc_id)

    metadata[file_name] = doc_ids
    save_faiss_index(index_file, documents_file, metadata_file, index, documents, metadata)
    return index, documents, metadata


def segment_text(text, max_length=512):
    sentences = text.split('. ')
    segments = []
    current_segment = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_length:
            segments.append('. '.join(current_segment) + '.')
            current_segment = []
            current_length = 0
        current_segment.append(sentence)
        current_length += sentence_length

    if current_segment:
        segments.append('. '.join(current_segment) + '.')
    return segments

def get_relevant_context(query, index, documents, top_k, embedding_model, max_context_length=1024):
    print(PINK + "Retrieving relevant context..." + RESET_COLOR)
    try:
        response = ollama.embeddings(model=embedding_model, prompt=query)
        query_embedding = np.array(response["embedding"]).astype('float32')
        if DEBUG:
            print(f"Query embedding with shape: {query_embedding.shape}")
        D, I = index.search(np.expand_dims(query_embedding, axis=0), top_k)
        
        context_segments = [documents[list(documents.keys())[i]] for i in I[0]]
        
        # Ensure the total length of the context does not exceed max_context_length
        ## TODO - Implement a more sophisticated method to combine context segments
        combined_context = ""
        current_length = 0
        for segment in context_segments:
            segment_length = len(segment.split())
            if current_length + segment_length > max_context_length:
                if current_length == 0:
                    combined_context += segment + "\n"  # Add at least one segment if all segments exceed the limit
                break
            combined_context += segment + "\n"
            current_length += segment_length

        return combined_context.strip()
    except Exception as e:
        print(f"Error getting relevant context: {str(e)}")
        return ""


def ollama_chat(user_input, system_message, index, documents, metadata, qa_model, embedding_model, conversation_history, top_k, client, document_content=None, index_file=None, documents_file=None, metadata_file=None, max_context_length=1024):
    if document_content:
        segments = segment_text(document_content)
        segment_embeddings = []
        segment_ids = []
        file_name = 'document_content'  
        for segment in segments:
            try:
                response = ollama.embeddings(model=embedding_model, prompt=segment)
                embedding = np.array(response["embedding"]).astype('float32')
                segment_embeddings.append(embedding)
                segment_ids.append(str(uuid.uuid4()))
            except Exception as e:
                print(f"Error generating embeddings for segment: {str(e)}")
                continue
        for seg_id, embedding in zip(segment_ids, segment_embeddings):
            index.add(np.expand_dims(embedding, axis=0))
            documents[seg_id] = segment
        metadata[file_name] = segment_ids
        save_faiss_index(index_file, documents_file, metadata_file, index, documents, metadata)
        relevant_context = get_relevant_context(user_input, index, documents, top_k, embedding_model, max_context_length)
        context_str = relevant_context
        if DEBUG:
            print(f"Context Pulled from Document using spesific document: \n\n{context_str}")
    else:
        relevant_context = get_relevant_context(user_input, index, documents, top_k, embedding_model, max_context_length)
        if relevant_context:
            context_str = relevant_context
            if DEBUG:
                print(f"Context Pulled from Documents using all documents: \n\n{context_str}")
        else:
            print("No relevant context found.")
            context_str = ""

    user_input_with_context = user_input
    if context_str:
        user_input_with_context = context_str + "\n\n" + " User Question: " + user_input

    conversation_history.append({"role": "user", "content": user_input_with_context})
    messages = [{"role": "system", "content": system_message}, *conversation_history]

    if DEBUG:
        print(f"Messages being sent to QA model: {messages}")

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



def list_documents(metadata):
    print(PINK + "Listing documents..." + RESET_COLOR)
    pdf_files = [file for file in metadata.keys() if file.endswith('.pdf')]
    return pdf_files



def get_document_by_id(documents, metadata, file_name):
    print(PINK + f"Retrieving document with name {file_name}..." + RESET_COLOR)
    if file_name in metadata:
        doc_ids = metadata[file_name]
        return "\n".join([documents[doc_id] for doc_id in doc_ids])
    else:
        return None


def clean_text_for_embedding(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

def clean_text_for_qa(text):
    text = ' '.join(text.split())
    return text
