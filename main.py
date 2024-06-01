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
import os 
from datetime import datetime
import re
class LLM:
    def __init__(self):
        df = pd.read_csv('students.csv')
        if not os.path.exists('keywords.csv'):
            empty_df = pd.DataFrame(data=[["테스트",0]],columns=['keyword','count'])  # You can specify column names here
            # Write the DataFrame to a CSV file
            empty_df.to_csv('keywords.csv', index=False)
        self.df_keyword = pd.read_csv('keywords.csv')
        
        self.ids = df[df['학번'] == 111111].to_dict(orient='records')
        aks = ["전북대학교 컴퓨터공학부 졸업 조건을 알려줘","전북대학교 인공지능학과 졸업 조건을 알려줘","전북대학교 인공지능학과랑 컴퓨터공학부의 차이를 알려줘", 
               "지금 휴학 가능한가요? 일반휴학입니다","3.14~15까지 하는 추가 추가 수강신청은 뭐야?","부전공 포기신청 기간이 따로 정해져있냐? 단대 행정실에 제출하려고했는데ㅜㅜ"
               ,"자퇴신청 어제 까진데 또 언제할수있음?","학교일정 어디 봄? 2024 전체요","계절학기 수강신청 5/7일 맞나요?","나의 학번을 알려줘"]
        # aks = ["나의 학번을 알려줘"]
        self.retriever = EmbeddingRetriever()
        self.content_processor = GPTAnswer()
    def generate_response(self,answer):
        query = answer
        output_format = "" # User can specify output format
        key_word_output_format = ""
        profile = "" # User can define the role for LLM
        print("Fetch web content based on the query")
        # Fetch web content based on the query
        web_contents_fetcher = WebContentFetcher("전북대 : " + query + str(datetime.now()))
        web_contents, serper_response = web_contents_fetcher.fetch()
        # Retrieve relevant documents using embeddings
        print("Retrieve relevant documents using embeddings 1-1")

        print("Retrieve relevant documents using embeddings 1-2")
        relevant_docs_list,links = self.retriever.retrieve_embeddings("전북대 : " + query + str(datetime.now()))
        print("Retrieve relevant documents using embeddings 1-3")
        print("Retrieve relevant documents using embeddings 1-4")
        formatted_relevant_docs = self.content_processor._format_reference(relevant_docs_list, links, web_contents, serper_response )
        print(formatted_relevant_docs)
        print("Measure the time taken to get an answer from the GPT model")
        # Measure the time taken to get an answer from the GPT model
        start = time.time()
        print("Generate answer from ChatOpenAI")
        # Generate answer from ChatOpenAI
        keyword = self.content_processor.get_keyword(query, "ko-KR", key_word_output_format, profile)
        if keyword in self.df_keyword['keyword'].values:
            count = int(self.df_keyword.loc[self.df_keyword['keyword'] == keyword]['count']) + 1
            self.df_keyword.loc[self.df_keyword['keyword'] == keyword,'count'] = count
            self.df_keyword.to_csv('keywords.csv',index=False)
        else:   
            new_df = pd.DataFrame(data=[[keyword,0]], columns=['keyword','count'])
            # self.df_keyword.append(new_df,ignore_index=True)
            pd.concat([self.df_keyword,new_df],ignore_index=True).to_csv('keywords.csv',index=False)
            # self.df_keyword.to_csv('keywords.csv')
        ai_message_obj = self.content_processor.get_answer(query, formatted_relevant_docs, "ko-KR", output_format, profile, self.ids)
        answer = ai_message_obj + '\n'
        # st.info(answer)
        end = time.time()
        print("\n\nGPT Answer time:", end - start, "s")
        return answer
    def checker(self, query):
        key_word_output_format = ""
        profile = "" # User can define the role for LLM
        keyword = self.content_processor.get_search("전북대 : " + query, "ko-KR", key_word_output_format, profile)
        return keyword
def print_markdown_from_file(file_path):
    df = pd.read_csv(file_path)
    df_sorted = df.sort_values(by='count')
    markdown_content = df['keyword'].to_markdown(index=False,)
    st.markdown(markdown_content)    
def hide_streamlit_header_footer():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
if __name__ == "__main__":
    st.set_page_config(page_title="전북대학교")
    # st.title("전북대학교")
    llm  = LLM()
    # with st.sidebar:
    #     print_markdown_from_file("keywords.csv")
    hide_streamlit_header_footer()
    st.logo("GPT4V2/JBNU_main3.png", icon_image="GPT4V2/JBNU_main2.png")
    pattern = r'\b\d+\b'
    # Replicate Credentials

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "도움이 필요하신가요?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "도움이 필요하신가요?"}]
    st.sidebar.button('전체 지우기', on_click=clear_chat_history)
    
    # User-provided prompt
    if prompt := st.chat_input("메시지를 입력하세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("<생성중...>"):
                key = llm.checker(prompt)
                key_int = re.findall(pattern,key)[0]
                try:
                    if int(key_int) == 1:
                        response = llm.generate_response(prompt)
                    else: 
                        response = key.split('\n')[2].split(':')[1]
                except:
                    response = llm.generate_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

    # streamlit run /home2/bazaarz/desktop/Llama3/jbchat/GPT4V2/main.py --server.fileWatcherType none