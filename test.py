from fetch_web_content import WebContentFetcher
from retrieval import EmbeddingRetriever
from llm_answer import GPTAnswer
# from local_llm_answer import GPTAnswer
# from local_llm_answerv1 import GPTAnswer
from locate_reference import ReferenceLocator
import time
import pandas as pd
import json
import streamlit as st
from tqdm import tqdm
pd.set_option('display.max_colwidth', None)
class LLM:
    def __init__(self):
        df = pd.read_csv('students.csv')
        self.ids = df[df['학번'] == 111111].to_dict(orient='records')
        self.retriever = EmbeddingRetriever()
        self.content_processor = GPTAnswer()
    def generate_response(self,answer):
        query = answer
        output_format = "" # User can specify output format
        key_word_output_format = ""
        profile = "" # User can define the role for LLM
        print("Fetch web content based on the query")
        # Fetch web content based on the query
        web_contents_fetcher = WebContentFetcher("전북대 : "+query)
        web_contents, serper_response = web_contents_fetcher.fetch()
        # Retrieve relevant documents using embeddings
        print("Retrieve relevant documents using embeddings 1-1")
        print("Retrieve relevant documents using embeddings 1-2")
        relevant_docs_list,links = self.retriever.retrieve_embeddings(query)
        print("Retrieve relevant documents using embeddings 1-3")
        print("Retrieve relevant documents using embeddings 1-4")
        formatted_relevant_docs = self.content_processor._format_reference(relevant_docs_list, links, web_contents, serper_response )
        print("Measure the time taken to get an answer from the GPT model")
        # Measure the time taken to get an answer from the GPT model
        start = time.time()
        print("Generate answer from ChatOpenAI")
        # Generate answer from ChatOpenAI
        ai_message_obj = self.content_processor.get_answer(query, formatted_relevant_docs, "ko-KR", output_format, profile, self.ids)
        answer = ai_message_obj + '\n'
        keyword = self.content_processor.get_keyword(query, "ko-KR", key_word_output_format, profile)
        end = time.time()
        return {"answer":answer,"keyword":keyword}
if __name__ == "__main__":
    dfQA = pd.read_csv("QA.csv")
    for i in [2,3]:
        df = pd.DataFrame(columns={f"Q{i}-3"})
        for id,text in tqdm(enumerate(dfQA[f"Q{i}-1"])):
            if pd.isna(text):
                break
            if id % 2 == 0:
                llm = LLM()
            try:
                answer =  llm.generate_response(str(text))
            except Exception as e:
                answer = str(e)
            df = df.append({f"Q{i}-3": answer},ignore_index=True)
        dfQA[f"Q{i}-3"] = df[f"Q{i}-3"]
    dfQA.to_csv("QA_answer_gpt4.csv", encoding='utf-8')