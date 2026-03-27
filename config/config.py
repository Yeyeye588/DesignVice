import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # OpenAI配置
    OPENAI_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    OPENAI_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

    # MySQL配置
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag")

    # Chroma配置
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "../core/chroma_db")

    # 其他配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    TOP_K = int(os.getenv("TOP_K", 5))
    EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "D:\Design\modelscope\hub\models\AI-ModelScope\bge-large-zh-v1.5")