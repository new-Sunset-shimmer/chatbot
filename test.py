from fetch_web_content import WebContentFetcher
from retrieval import EmbeddingRetriever
from llm_answer import GPTAnswer
from locate_reference import ReferenceLocator
import time
import pandas as pd
import json
import streamlit as st
class LLM:
    def __init__(self):
        df = pd.read_csv('students.csv')
        self.ids = df[df['학번'] == 111111].to_dict(orient='records')
        self.aks = ["전북대학교 컴퓨터공학부 졸업 조건을 알려줘","전북대학교 인공지능학과 졸업 조건을 알려줘","전북대학교 인공지능학과랑 컴퓨터공학부의 차이를 알려줘", 
               "지금 휴학 가능한가요? 일반휴학입니다","3.14~15까지 하는 추가 추가 수강신청은 뭐야?","부전공 포기신청 기간이 따로 정해져있냐? 단대 행정실에 제출하려고했는데ㅜㅜ"
               ,"자퇴신청 어제 까진데 또 언제할수있음?","학교일정 어디 봄? 2024 전체요","계절학기 수강신청 5/7일 맞나요?"]
        # aks = ["나의 학번을 알려줘"]
        self.retriever = EmbeddingRetriever()
        self.content_processor = GPTAnswer()
    def generate_response(self,answer):
        query = answer
        output_format = "" # User can specify output format
        profile = "" # User can define the role for LLM
        print("Fetch web content based on the query")
        # Fetch web content based on the query
        # web_contents_fetcher = WebContentFetcher(query)
        # web_contents, serper_response = web_contents_fetcher.fetch()
        # Retrieve relevant documents using embeddings
        print("Retrieve relevant documents using embeddings 1-1")

        print("Retrieve relevant documents using embeddings 1-2")
        relevant_docs_list,links = self.retriever.retrieve_embeddings(query)
        print("Retrieve relevant documents using embeddings 1-3")
        print("Retrieve relevant documents using embeddings 1-4")
        formatted_relevant_docs = self.content_processor._format_reference(relevant_docs_list, links)
        print(formatted_relevant_docs)
        print("Measure the time taken to get an answer from the GPT model")
        # Measure the time taken to get an answer from the GPT model
        start = time.time()
        print("Generate answer from ChatOpenAI")
        # Generate answer from ChatOpenAI
        ai_message_obj = self.content_processor.get_answer(query, formatted_relevant_docs, "ko-KR", output_format, profile, self.ids)
        answer = ai_message_obj + '\n'
        return answer
        end = time.time()
        print("\n\nGPT Answer time:", end - start, "s")
if __name__ == "__main__":
    asks = [["전북대학교 컴퓨터공학부 졸업 조건을 알려줘","전북대학교 인공지능학과 졸업 조건을 알려줘","전북대학교 인공지능학과랑 컴퓨터공학부의 차이를 알려줘"], 
               ["지금 휴학 가능한가요? 일반휴학입니다","3.14~15까지 하는 추가 추가 수강신청은 뭐야?","부전공 포기신청 기간이 따로 정해져있냐? 단대 행정실에 제출하려고했는데ㅜㅜ"]
               ,["자퇴신청 어제 까진데 또 언제할수있음?","학교일정 어디 봄? 2024 전체요","계절학기 수강신청 5/7일 맞나요?"]]
    df = pd.DataFrame({"ask":[],"answer":[]})
    for texts in asks:
        llm = LLM()
        for text in texts:
            answer =  llm.generate_response(text)
            df = df.append({"ask": text, "answer": answer},ignore_index=True)
        time.sleep(60)
    df.to_csv("QA_test.csv")
    # Optional Part: display the reference sources of the quoted sentences in LLM's answer
    # 
    # print("\n\n", "="*30, "Refernece Cards: ", "="*30, "\n")
    # locator = ReferenceLocator(answer, serper_response)
    # reference_cards = locator.locate_source()
    # json_formatted_cards = json.dumps(reference_cards, indent=4)
    # print(json_formatted_cards)