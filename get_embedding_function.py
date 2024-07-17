from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

def get_embedding_function():
    #embeddings = BedrockEmbeddings()
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
