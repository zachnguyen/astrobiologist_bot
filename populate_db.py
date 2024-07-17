import argparse
import os, json
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "data/chroma"
DATA_SOURCE_PATH = "transcripts"
with open('transcripts/video_metadata.json', 'r') as json_file:
    metadata = json.load(json_file)

def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    

def load_documents():
    loader = DirectoryLoader(DATA_SOURCE_PATH, glob="*.md")
    documents = loader.load()
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )

    print(f'''Number of docs: {len(db.get(include=[])['ids'])}''')

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print('DONE ADDING')
        
        #db.persist()
    else:
        print("✅ No new documents to add")
    


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/isaacarthurSFIA_692.txt:0"
    # Source filename : Chunk Index

    last_source_id = None
    current_chunk_index = 0

    for chunk in chunks:
        current_source_id = chunk.metadata.get("source")
        #print(current_source_id)

        # If the Source ID is the same as the last one, increment the index.
        if current_source_id == last_source_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Add it to the meta-data.
        chunk_id = f"{current_source_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        chunk.metadata["file_name"] = current_source_id
        chunk.metadata["vid_title"] = metadata[current_source_id[len("transcripts/"):-len('.md')]]["video_title"]
        chunk.metadata["published_dt"] = metadata[current_source_id[len("transcripts/"):-len('.md')]]["published_dt"]
        chunk.metadata["host_name"] = metadata[current_source_id[len("transcripts/"):-len('.md')]]["host"]
        
        last_source_id = current_source_id
        #print(chunk)

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
