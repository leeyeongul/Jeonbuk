from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
import gdown
import zipfile
import streamlit as st
import numpy as np

# Streamlit Secrets에서 API 키 불러오기
st_api_key = st.secrets["OPENAI_API_KEY"]

class psds_callcenter :
    def __init__(self, model="text-embedding-3-large", path='faiss_index') :

        if not os.path.exists(path):
            # Replace this with your Google Drive file ID
            file_id = "1T_ptoHBEuKe3j0iTQ7mymXS2YaY72YYt"

            # Build the Google Drive URL
            url = f"https://drive.google.com/uc?id={file_id}"

            # Destination path for the downloaded file
            destination = "faiss_index.zip"

            # Download the file
            gdown.download(url, destination, quiet=False)

            # Path to the .zip file
            zip_file_path = "faiss_index.zip"

            # Destination folder
            destination_folder = "faiss_index"

            # Ensure the destination folder exists
            os.makedirs(destination_folder, exist_ok=True)

            # Extract the .zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(destination_folder)
                
        # load_dotenv()  # important
        # self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=st_api_key)  # this is also the default, it can be omitted
        self.embedding_model = OpenAIEmbeddings(openai_api_key=st_api_key, model=model)
        self.vectorstore = FAISS.load_local(path, embeddings=self.embedding_model, allow_dangerous_deserialization=True)
        
    def rag_output(self, query, k=5, model='gpt-4-turbo', cosine_threshold=0.5) :
        self.query = query
        self.cosine_threshold = cosine_threshold
        query_embedding = self.embedding_model.embed_query(self.query)
        def normalize_vector(vector):
            return vector / np.linalg.norm(vector)
        
        # 쿼리 벡터 정규화
        normalized_query_embedding = normalize_vector(query_embedding)
        retrieved_docs = self.vectorstore.similarity_search_with_score_by_vector(normalized_query_embedding, k=k)
        def create_prompt(query, retrieved_docs, cosine_threshold=0.5):
            self.context = ""
            for doc in retrieved_docs:
                cosine_score = doc[-1]
                if cosine_score > cosine_threshold :
                    self.context += f"출처: {doc[0].metadata['source']}\n"
                    self.context += f"내용: {doc[0].page_content}\n\n"
        
            prompt = f"""
        아래는 사용자가 요청한 정보와 관련된 문서 내용입니다:
        {self.context}
        사용자의 질문에 따라 응답을 생성하세요: "{self.query}"
        """
            return prompt
        self.prompt = create_prompt(query,retrieved_docs, cosine_threshold=self.cosine_threshold)
        messages = [
        {"role": "user", "content": f"""{self.prompt}"""}
        ]
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            n=1,
            stop=None,
            temperature=0.7
        )
        self.generated_response = response.choices[0].message.content
            
        def summaize_context(context) :
            sumarize_prompt = f'''
        아래 내용을 보기 좋게 정리해줘
        {context}
        '''
            summarize_messages = [
            {"role": "user", "content": f"""{sumarize_prompt}"""}
            ]
            summarize_response = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=summarize_messages,
                max_tokens=4096,
                n=1,
                stop=None,
                temperature=0.7
            )
            summarize_response = summarize_response.choices[0].message.content
            return summarize_response
            
        self.summarized_source = summaize_context(self.context)
        print("Generated Answer:")
        print(self.generated_response)
        print("\nSources:")
        print(self.summarized_source)
        self.final_output = 'generated_response:' +'\n\n'+ self.generated_response + '\n\n'+ 'Sources:' +'\n\n' + self.summarized_source
        return self.final_output

rag = psds_callcenter()

st.title("RAG-Based 전북 콜센터 Generator")

query = st.text_input("Enter your query:")

if st.button("Generate"):
    final_output = rag.rag_output(query=query, cosine_threshold=0.65)  # k는 참조할 사례 갯수
    st.text_area("전북콜센터 상담:", value=final_output, height=400)