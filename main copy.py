from fetch_web_content import WebContentFetcher
from retrieval import EmbeddingRetriever
from llm_answer import GPTAnswer
from locate_reference import ReferenceLocator
import time
import pandas as pd
import json
if __name__ == "__main__":
    df = pd.read_csv('students.csv')
    ids = df[df['학번'] == 111111].to_dict(orient='records')
    aks = ["전북대학교 컴퓨터공학부 졸업 조건을 알려줘","전북대학교 인공지능학과 졸업 조건을 알려줘","전북대학교 인공지능학과랑 컴퓨터공학부의 차이를 알려줘"]
    # aks = ["나의 학번을 알려줘"]
    retriever = EmbeddingRetriever()
    content_processor = GPTAnswer()
    for i in range(3):
        query = aks[i]
        output_format = "" # User can specify output format
        profile = "" # User can define the role for LLM
        print("Fetch web content based on the query")
        # Fetch web content based on the query
        # web_contents_fetcher = WebContentFetcher(query)
        # web_contents, serper_response = web_contents_fetcher.fetch()
        # Retrieve relevant documents using embeddings
        print("Retrieve relevant documents using embeddings 1-1")

        print("Retrieve relevant documents using embeddings 1-2")
        relevant_docs_list,links = retriever.retrieve_embeddings(query)
        print("Retrieve relevant documents using embeddings 1-3")
        print("Retrieve relevant documents using embeddings 1-4")
        formatted_relevant_docs = content_processor._format_reference(relevant_docs_list, links)
        print(formatted_relevant_docs)
        print("Measure the time taken to get an answer from the GPT model")
        # Measure the time taken to get an answer from the GPT model
        start = time.time()
        print("Generate answer from ChatOpenAI")
        # Generate answer from ChatOpenAI
        ai_message_obj = content_processor.get_answer(query, formatted_relevant_docs, "ko-KR", output_format, profile, ids)
        answer = ai_message_obj + '\n'
        end = time.time()
        print("\n\nGPT Answer time:", end - start, "s")

    # Optional Part: display the reference sources of the quoted sentences in LLM's answer
    # 
    # print("\n\n", "="*30, "Refernece Cards: ", "="*30, "\n")
    # locator = ReferenceLocator(answer, serper_response)
    # reference_cards = locator.locate_source()
    # json_formatted_cards = json.dumps(reference_cards, indent=4)
    # print(json_formatted_cards)