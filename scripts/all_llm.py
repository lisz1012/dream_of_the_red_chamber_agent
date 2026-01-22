from langchain_openai import ChatOpenAI
from utils.env_utils import OPENAI_API_KEY, OPENAI_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

llm_gpt_4o_mini = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0.8,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    # model_kwargs={"response_format": {"type": "json_object"}}
)

llm_deepseek_v3 = ChatOpenAI(
    model="deepseek-chat",              # 或者 “deepseek-reasoner”/其他名字，依据 API 文档
    temperature=0.8,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    # 如果需要，还可传 model_kwargs 比如 max_tokens, context_length 等
)