## Author: Steffen Rogne
## Date: 19-05-2024
## Project: Own version of RAG with OLLAMA


import argparse
from helper import load_config, initialize_faiss, open_file, load_or_generate_embeddings, ollama_chat, list_documents, get_document_by_id
from openai import OpenAI

PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def main():
    parser = argparse.ArgumentParser(description="CLI tool for managing and querying PDF files with embeddings.")
    parser.add_argument("--config", type=str, default="./config.yaml", help="Path to the configuration file.")
    parser.add_argument("--upload", type=str, help="Path to the PDF file to upload.")
    parser.add_argument("--query", type=str, help="Query string to ask the model.")
    parser.add_argument("--list", action='store_true', help="List available documents in the database.")
    parser.add_argument("--doc", type=str, help="Document ID to use for the query.")
    args = parser.parse_args()

    config = load_config(args.config)
    embedding_dim = 1024  
    index, documents, metadata = initialize_faiss(config['faiss']['index_file'], config['faiss']['documents_file'], config['faiss']['metadata_file'], embedding_dim)
    
    if args.list:
        docs = list_documents(metadata)
        print(CYAN + "Available documents:" + RESET_COLOR)
        for doc in docs:
            print(f"- {doc}")
        return

    if args.upload:
        vault_content = open_file(args.upload)
        if vault_content:
            load_or_generate_embeddings(index, documents, metadata, args.upload, [vault_content], config['embedding_model'], config['faiss']['index_file'], config['faiss']['documents_file'], config['faiss']['metadata_file'])
            
    if args.query:
        if args.doc:
            document_content = get_document_by_id(documents, metadata, args.doc)
            if not document_content:
                print(f"Document with ID '{args.doc}' not found.")
                return
        else:
            document_content = None

        system_message = config['system_message']
        qa_model = config['qa_model']
        embedding_model = config['embedding_model']
        conversation_history = []
        client = OpenAI(base_url=config['ollama_api']['base_url'], api_key=config['ollama_api']['api_key'])

        answer = ollama_chat(args.query, system_message, index, documents, metadata, qa_model, embedding_model, conversation_history, config['top_k'], client, document_content, config['faiss']['index_file'], config['faiss']['documents_file'], config['faiss']['metadata_file'], config['max_context_length'])
        print(NEON_GREEN + "Answer: \n" + answer + RESET_COLOR)

if __name__ == "__main__":
    main()



