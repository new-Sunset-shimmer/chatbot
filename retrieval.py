import yaml
import os
from fetch_web_content import WebContentFetcher
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.retrievers import (
    ElasticSearchBM25Retriever,
)


class EmbeddingRetriever:
    TOP_K = 5  # Number of top K documents to retrieve

    def __init__(self):
        # Load configuration from config.yaml file
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )

    def retrieve_embeddings(self,query: str):
        # Retrieve embeddings for a given list of contents and a query
        # metadatas = [{'url': link} for link in link_list]
        # texts = self.text_splitter.create_documents(contents_list, metadatas=metadatas)


        # Create a Chroma database from the documents using specific embeddings
        print("chroma")
        db = Chroma(persist_directory="./jbnu_db", 
                    embedding_function= SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask"
                                                                      ,model_kwargs = {'device': 'cuda'}),)
        print("after chroma")
        # Create a retriever from the database to find relevant documents
        retriever = db.as_retriever(search_kwargs={"k": self.TOP_K})
        result = retriever.get_relevant_documents(query)
        links = [link.metadata['url'] for id,link in enumerate(result)]
        return result,links # Retrieve and return the relevant documents

# Example usage
if __name__ == "__main__":
    query = "전북대학교 컴퓨터공학부 졸업조건을 알려줘"

    # Create a WebContentFetcher instance and fetch web contents
    # Create an EmbeddingRetriever instance and retrieve relevant documents
    retriever = EmbeddingRetriever()
    relevant_docs_list = retriever.retrieve_embeddings(query)

    print("\n\nRelevant Documents from VectorDB:\n", relevant_docs_list)
    