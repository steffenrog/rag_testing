# PDF Management and Querying Tool

This project is a CLI tool for managing and querying PDF files with embeddings. It allows you to upload PDFs, query them, and manage the document database. The project use two AI models, one for handeling embeddings, and one for handeling query. After testing different models, the phi3 with 128k token context seems like a good option for QA. 

--This project runs offline as it is not dependend on langchein or any web services 

## Installation

To get started with this project, you need to install and test Ollama.

1. **Install Ollama**:
    ```sh
    # Ensure you have Ollama installed
    For mac you can use "brew install ollama"
    Other OS refer to the webpage "https://ollama.com"
    pip install ollama
    ```

2. **Pull the required model**:
    ```sh
    # Pull the necessary model from Ollama
    ollama pull mxbai-embed-large
    ollama pull phi3:14b-medium-128k-instruct-q4_0
    ```

3. **Run a test to verify the installation**:
    ```sh
    # Run a test to ensure everything is set up correctly
    ollama run mxbai-embed-large
    ollama run phi3:14b-medium-128k-instruct-q4_0
    use ctrl+d to exit
    ```
 4. **Install requirements**:
    ```sh
    conda create rag python=3.10.9
    pip install -r requirements.txt
    # Install torch
    conda install pytorch-nightly::pytorch torchvision torchaudio -c pytorch-nightly
    # or look at https://pytorch.org/get-started/locally/
   ```

## Running the Main Script

The main script, `main.py`, is designed to be a versatile tool for managing and querying your PDF documents. Below are the available commands and their descriptions.

### Command Line Arguments

- `--config`: Path to the configuration file.
  - Default: `./config.yaml`
- `--upload`: Path to the PDF file to upload.
- `--query`: Query string to ask the model.
- `--list`: List available documents in the database.
- `--doc`: Document ID to use for the query.

### Example Usage

1. **Uploading a PDF**:
    ```sh
    python main.py --upload path/to/your/file.pdf
    ```

2. **Querying the Model**:
    ```sh
    python main.py --query "Your query string here"
    ```

3. **Listing Available Documents**:
    ```sh
    python main.py --list
    ```

4. **Querying a Specific Document**:
    ```sh
    python main.py --query "Your query string here" --doc document_id
    ```

5. **Specifying a Configuration File**:
    ```sh
    python main.py --config path/to/your/config.yaml
    ```

Ensure that you have all dependencies installed and correctly configured as per your `config.yaml` file.

Happy querying!
