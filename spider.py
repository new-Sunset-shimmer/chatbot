import yaml
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.retrievers import BM25Retriever # pip install rank_bm25
import requests
from bs4 import BeautifulSoup
from collections import deque
import re
from urllib.parse import urlparse
import urllib
from web_crawler import WebScraper
import threading
import time
from langchain.docstore.document import Document
from langchain_elasticsearch import ElasticsearchStore
import pandas as pd
class WebScraper:
    def __init__(self, user_agent='macOS'):
        # Initialize the scraper with a user agent (default is 'macOS')
        self.headers = self._get_headers(user_agent)
        self.exclude_domains = ['facebook.com', 'twitter.com', 'x.com', 'kakao.com', "#", "instagram.com", "youtube.com", "youtu.be"]
        self.retriever = EmbeddingRetriever()
        self.web_contents = []  # Stores the fetched web contents
        self.error_urls = []  # Stores URLs that resulted in an error during fetching
        self.web_contents_lock = threading.Lock()  # Lock for thread-safe operations on web_contents
        self.error_urls_lock = threading.Lock()  # Lock for thread-safe operations on error_urls
    def is_same_domain(self, url, domains):
        parsed_url = urlparse(url)
        return parsed_url.netloc in domains

    def _get_headers(self, user_agent):
        # Private method to get headers for the request based on the specified user agent
        if user_agent == 'macOS':
            # Headers for macOS user agent
            return {
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"macOS"',
            }
        else:
            # Headers for Windows user agent
            return {
                'Upgrade-Insecure-Requests': '1',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
                'sec-ch-ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
            }

    def get_webpage_html(self, url):
        # Fetch the HTML content of a webpage from a given URL
        response = requests.Response()  # Create an empty Response object
        if url.endswith(".pdf"):
            # Skip PDF files which are time consuming
            return response

        try:
            # Attempt to get the webpage content with specified headers and timeout
            response = requests.get(url, headers=self.headers, timeout=8)
            response.encoding = "utf-8"
        except requests.exceptions.Timeout:
            # Add timeout exception handling here
            return response
        
        return response

    def convert_html_to_soup(self, html):
        # Convert the HTML string to a BeautifulSoup object for parsing
        html_string = html.text
        return BeautifulSoup(html_string, "lxml")

    def extract_main_content(self, html_soup, rule=0):
        # Extract the main content from a BeautifulSoup object
        main_content = []
        # tag_rule = re.compile("^(h[1-6]|p|div|span|th|caption)" if rule == 1 else "^(h[1-6]|p)")
        # # Iterate through specified tags and collect their text
        # for tag in html_soup.find_all(tag_rule):
        #     tag_text = tag.get_text().strip()
        #     if tag_text and len(tag_text.split()) > 2:
                # main_content.append(tag_text)
        main_content = html_soup.get_text()
        return "\n".join(main_content).strip()
    def scrape_page(self, start_url, rule=0, max_pages=None):
        visited = set()
        queue = deque([(start_url, 0)])  # Queue now contains tuples (url, depth)
        
        # Extract main content or process the page as needed
        # titles = soup.find('meta', property='og:title')['content']
        depth_check = 0
        while queue:
            current_url, depth = queue.popleft()
            try:
                if current_url in visited:
                    continue
                # Fetch the webpage HTML
                # print(depth)
                if depth > depth_check:
                    self.retriever.htmls.to_csv('htmls.csv', index=False)
                    depth_check = depth
                webpage_html = self.get_webpage_html(current_url)
                soup = self.convert_html_to_soup(webpage_html)
                main_content = self.extract_main_content(soup, rule)
                # Extract main content or process the page as needed
                titles = soup.find('meta', property='og:title')['content']
                self.retriever.retrieve_embeddings(titles + main_content, current_url)
                # Add links to the queue
                follow_links = soup.find_all('a', href=True)
                for link in follow_links:
                    follow_url = link['href']
                    if (follow_url.startswith('#') or self.is_same_domain(follow_url, self.exclude_domains) 
                        or re.search("download.do", follow_url)):  # Skip anchor links
                        continue
                    if not follow_url.startswith(('http://', 'https://')):  # Handle relative links
                        follow_url = urllib.parse.urljoin(current_url, follow_url)
                    queue.append((follow_url, depth + 1))  # Increment depth for the next level
                visited.add(current_url)
                if depth >= max_pages:
                    print("Reached maximum number of pages to crawl.")
                    break

            except Exception as e:
                # print(f"Error occurred while scraping {current_url}: {e}")
                pass
        print("Crawling completed.")
# Example usage    
class EmbeddingRetriever:
    TOP_K = 10  # Number of top K documents to retrieve

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
        embed = embedding_function=SentenceTransformerEmbeddings(model_name="jhgan/ko-sroberta-multitask",model_kwargs = {'device': 'cuda'})
        self.Chromadb = Chroma(
            persist_directory="./jbnu_db_all_text",
            embedding_function=embed,)
        # self.FAISSdb = FAISS(
        #     persist_directory="./jbnu_db_FAISS",
        #     embedding_function=embed)
        self.htmls = pd.read_csv('htmls.csv')
        self.htmls_temp = {"url":[],"content":[]}

    def retrieve_embeddings(self, content: str, link: str):
        # Retrieve embeddings for a given list of contents and a query
        metadatas = [{'url': link}]
        text = self.text_splitter.create_documents([content], metadatas)
        # Create a Chroma database from the documents using specific embeddings
        self.Chromadb.add_documents(
            text,
            # SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",model_kwargs = {'device': 'cuda'}),
            
            # SentenceTransformerEmbeddings(model_name="BM-K/KoSimCSE-roberta-multitask",model_kwargs = {'device': 'cuda'}),
        )
        # self.htmls_temp["url"].append(link)
        # self.htmls_temp["content"].append(content)
        self.htmls.append({"url":[link],"content":[content]}, ignore_index=False)
        # self.FAISSdb.add_documents(
        #     text,
        #     # SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",model_kwargs = {'device': 'cuda'}),
            
        #     # SentenceTransformerEmbeddings(model_name="BM-K/KoSimCSE-roberta-multitask",model_kwargs = {'device': 'cuda'}),
        # )
        

# Example usage
if __name__ == "__main__":
    scraper = WebScraper(user_agent='macOS')
    test_url = ["http://www.jbnu.ac.kr/kor/?menuID=139","https://csai.jbnu.ac.kr/csai/29107/subview.do","https://cse.jbnu.ac.kr/cse/index.do"]
    for url in test_url:
        scraper.scrape_page(url, max_pages=5)  # Specify the max