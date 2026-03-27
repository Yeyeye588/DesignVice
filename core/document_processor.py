


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
import os
import tempfile



class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

        # 父子文档分割器
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE * 2,  # 父文档更大
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？"]
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE // 2,  # 子文档更小
            chunk_overlap=self.config.CHUNK_OVERLAP // 2,
            separators=["\n", "。", "！", "？", ".", "!", "?", " "]
        )

    def  load_document(self, uploaded_file):
        """加载上传的文档"""
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        print(tmp_path)
        try:
            # 根据文件类型选择加载器
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            elif uploaded_file.name.endswith('.docx'):
                loader = Docx2txtLoader(tmp_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_path, encoding='utf-8')
            else:
                raise ValueError(f"不支持的文件类型: {uploaded_file.name}")
            # 加载的文档数据
            documents = loader.load()
            return documents
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


    def create_parent_child_chunks(self, documents, fileName):
        """创建父子文档块"""
        parent_docs = self.parent_splitter.split_documents(documents)
        child_docs = []

        for i, parent_doc in enumerate(parent_docs):
            # 为每个父文档创建唯一ID
            parent_id = f"{fileName}_parent_{i}"
            parent_doc.metadata['parent_id'] = parent_id
            parent_doc.metadata['doc_type'] = 'parent'

            # 从父文档创建子文档
            child_chunks = self.child_splitter.split_documents([parent_doc])
            for j, child_doc in enumerate(child_chunks):
                child_doc.metadata['parent_id'] = parent_id
                child_doc.metadata['child_id'] = f"child_{i}_{j}"
                child_doc.metadata['doc_type'] = 'child'
                child_docs.append(child_doc)


        # 获取到父文档 子文档
        # 父文档   documents(metadata={'parent_id': 'aaa.txt_parent_0', 'doc_type': "parent"}, page_conten='文档内容')
        # 子文档   documents(metadata={'parent_id': 'aaa.txt_parent_0', 'doc_type': "child", 'child_id': 'child_0_0'}, page_conten='文档内容')

        return parent_docs, child_docs
