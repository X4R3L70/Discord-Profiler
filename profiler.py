import os
import glob
import json
import warnings
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

PERSIST_DIR = "./storage"

def load_chat_logs_as_documents(file_path):
    # This function is unchanged
    documents = []
    file_name = os.path.basename(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not (isinstance(data, dict) and 'messages' in data):
        return []
    for message in data['messages']:
        author_name = "Unknown"
        if (isinstance(message.get('author'), dict) and message['author'].get('name')):
            author_name = message['author']['name']
        content = message.get('content', '')
        if content:
            doc_text = f"Author: {author_name}\nContent: {content}"
            document = Document(
                text=doc_text,
                metadata={"author": author_name, "file_source": file_name, "timestamp": message.get('timestamp')}
            )
            documents.append(document)
    return documents

def get_unique_authors(data_path):
    # This function is unchanged
    print("Identifying unique authors from JSON files...")
    authors = set()
    json_files = glob.glob(os.path.join(data_path, "*.json"))
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'messages' in data:
                for msg in data['messages']:
                    if 'author' in msg and 'name' in msg['author']:
                        authors.add(msg['author']['name'])
        except Exception as e:
            print(f" > Could not process {os.path.basename(file_path)} for authors: {e}")
    author_list = sorted(list(authors))
    print(f"Found {len(author_list)} unique authors.")
    return author_list

# --- NEW: Helper function to parse user's selection ---
def parse_selection(selection_str, authors):
    """Parses a user's selection string (e.g., '1, 3-5') into a list of author names."""
    selected_indices = set()
    parts = selection_str.split(',')
   
    for part in parts:
        part = part.strip()
        if not part: continue
       
        try:
            if '-' in part:
                start, end = map(int, part.split('-'))
                if start > end or start < 1 or end > len(authors):
                    print(f"Warning: Invalid range '{part}'. Skipping.")
                    continue
                # Add all indices from start to end (inclusive)
                for i in range(start, end + 1):
                    selected_indices.add(i - 1)
            else:
                index = int(part)
                if index < 1 or index > len(authors):
                    print(f"Warning: Invalid number '{part}'. Skipping.")
                    continue
                selected_indices.add(index - 1)
        except ValueError:
            print(f"Warning: Invalid input '{part}'. Skipping.")
            continue
           
    # Return a list of author names based on the selected indices
    return [authors[i] for i in sorted(list(selected_indices))]

def main():
    """
    Main function with an interactive prompt for author selection.
    """
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.llm = Ollama(model="llama3", request_timeout=360.0)
    json_data_path = os.path.expanduser("~/serveur/json")
   
    # --- Index loading is unchanged ---
    if not os.path.exists(PERSIST_DIR):
        print("Index not found. Please run the script once to build the index first.")
        # The full indexing logic is omitted here for brevity but should be present in your file.
        return
    else:
        print(f"Loading existing index from '{PERSIST_DIR}'...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        print("Index loaded successfully.")
   
    if 'index' in locals() and index:
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=15)
        unique_authors = get_unique_authors(json_data_path)
        if not unique_authors:
            print("No authors found. Cannot perform analysis.")
            return

        # --- NEW: Interactive selection menu ---
        print("\nPlease select the authors you want to analyze:")
        for i, author in enumerate(unique_authors):
            print(f"  {i+1}. {author}")
       
        print("\nEnter numbers (e.g., 1, 3), ranges (e.g., 5-8), or type 'all'.")
       
        authors_to_analyze = []
        while not authors_to_analyze:
            user_input = input("Selection: ").strip()
            if user_input.lower() == 'all':
                authors_to_analyze = unique_authors
                break
           
            authors_to_analyze = parse_selection(user_input, unique_authors)
            if not authors_to_analyze:
                print("Your selection was empty or invalid. Please try again.")
       
        # --- Analysis loop now uses the selected list ---
        print(f"\nStarting analysis for {len(authors_to_analyze)} selected author(s)...")
        for author in authors_to_analyze:
            analysis_prompt = (
                f"You are a professional mentalist. From the provided chat logs, establish a detailed personality description "
                f"of the chat member named '{author}'. Focus only on this person. "
                f"Describe their likely interests, philosophical alignment, political alignment, relationships with other members, "
                f"and any obsessions. Cite specific examples from their writing to support your analysis."
            )
            print("\n" + "="*80)
            print(f"|| Analyzing: {author}")
            print("="*80)
            streaming_response = query_engine.query(analysis_prompt)
            print(f"\n|| Llama 3 Mentalist Analysis for {author}:")
            streaming_response.print_response_stream()
            print("\n")
        print("All analyses complete.")
    else:
        print("Could not load or create the index. Exiting.")

if __name__ == "__main__":
    main()
