import os

from dotenv import load_dotenv
from langchain_classic.chains.hyde.prompts import web_search
from langchain_community.tools import TavilySearchResults

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')
LOCAL_BASE_URL = os.getenv('LOCAL_BASE_URL')
ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY')

MILVUS_URI = 'http://1.95.116.112:19530'

COLLECTION_NAME = 't_collection01'
# MILVUS_URI = "/Users/shuzheng/PyCharmProjects/milvus_code/milvus_lite.db"
MILVUS_URI = "tcp://192.168.1.91:19530"
COLLECTION_NAME = 't_collection01'
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
web_search_tool = TavilySearchResults(max_results=1)