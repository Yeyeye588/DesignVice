

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI



class VectorStore:
    def __init__(self, config):
        self.config = config
        # 初始化模型
        embed_path = r"D:\Design\modelscope\hub\models\AI-ModelScope\bge-large-zh-v1.5"
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_path)

        # 初始化Chroma客户端
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR
        )

        # 父文档和子文档使用不同的集合
        self.parent_vectorstore = Chroma(
            client=self.chroma_client,
            collection_name="parent_documents",
            embedding_function=self.embeddings
        )

        self.child_vectorstore = Chroma(
            client=self.chroma_client,
            collection_name="child_documents",
            embedding_function=self.embeddings
        )

        # 初始化上下文压缩器
        self.llm = ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            model="qwen-plus-2025-04-28",
            temperature=0
        )
        self.compressor = LLMChainExtractor.from_llm(self.llm)

    def add_documents(self, parent_docs, child_docs, document_id):
        """添加文档到向量存储"""
        # 为所有文档添加document_id元数据
        for doc in parent_docs + child_docs:
            doc.metadata['document_id'] = str(document_id)
        # 存储父文档和子文档
        parent_ids = self.parent_vectorstore.add_documents(parent_docs)
        child_ids = self.child_vectorstore.add_documents(child_docs)
        # print(parent_ids, child_ids)
        return parent_ids, child_ids

    def create_retriever(self, use_compression=True):
        """创建检索器"""
        # 创建子文档检索器（用于初始检索）
        child_retriever = self.child_vectorstore.as_retriever(
            search_kwargs={
                "k": self.config.TOP_K * 2,  # 获取更多子文档
            }
        )
        if use_compression:
            # 使用上下文压缩检索器
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=child_retriever
            )
            return compression_retriever
        else:
            return child_retriever

    def get_parent_documents(self, child_docs):
        """根据子文档获取对应的父文档"""
        parent_ids = set()
        for doc in child_docs:
            if 'parent_id' in doc.metadata:
                parent_ids.add(doc.metadata['parent_id'])

        return self.get_parent_documents_by_metadata(list(parent_ids))


    def get_parent_documents_by_metadata(self, parent_ids):
        """根据parent_id列表获取父文档"""
        if not parent_ids:
            return []

        parent_docs = []
        for parent_id in parent_ids:
            try:
                # 使用相似度搜索并过滤parent_id
                results = self.parent_vectorstore.get(where={"parent_id": parent_id})
                parent_docs.extend(results['documents'][0])  # 每个parent_id只取一个结果
            except Exception as e:
                print(f"获取父文档时出错 (parent_id: {parent_id}): {e}")
                continue

        return parent_docs

