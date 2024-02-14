import json
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
import pprint
import os
from getpass import getpass

class FloodsQASystem:
    def __init__(self, file_path):
        self.file_path = file_path
        self.docs = self.load_data()

        texts, data = self.preprocess_data()

        self.embeddings = self.create_embeddings()

        self.vector_index = self.create_vector_index(texts)

        self.store = self.create_store(data)
        self.persist_store()

        self.setup_openai_api_key()
        
        self.qa_with_source = self.create_qa_chain()

        self.llm_chain = self.create_llm_chain()

        self.hist = []

    def setup_openai_api_key(self):
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    def load_data(self):
        loader = CSVLoader(file_path=self.file_path)
        return loader.load()

    def preprocess_data(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        context = "\n\n".join(str(p.page_content) for p in self.docs)
        texts = text_splitter.split_text(context)
        data = text_splitter.split_documents(self.docs)
        return texts, data

    def create_embeddings(self):
        return OpenAIEmbeddings()

    def create_vector_index(self, texts):
        return Chroma.from_texts(texts, self.embeddings).as_retriever()

    def create_store(self, data):
        return Chroma.from_documents(
            data,
            self.embeddings,
            ids=[f"{item.metadata['source']}-{index}" for index, item in enumerate(data)],
            collection_name="Floods",
            persist_directory='/Users/sarthakgupta/Downloads/db',
        )

    def persist_store(self):
        self.store.persist()

    def create_prompt_template(self, template, input_variables):
        return PromptTemplate(template=template, input_variables=input_variables)

    def create_qa_chain(self):
        template = """You are a bot that answers questions about floods, using only the context provided.
        The context provided is from a wikipedia page on floods which contains information about all
        the floods.
        If you don't know the answer, simply state that you don't know.

        {context}

        Question: {question}"""

        PROMPT = self.create_prompt_template(
            template=template, input_variables=["context", "question"]
        )

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.store.as_retriever(),
            chain_type_kwargs={"prompt":PROMPT,"verbose":True},
            return_source_documents=True,
        )

    def create_chat_openai(self, temperature, model):
        return ChatOpenAI(temperature=temperature, model=model)

    def create_llm_chain(self):
        template_summarize = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone 
        question. The question should contain as much information as possible about the context.

        Chat History:
        {chat_history}
        Follow Up Input: {query}
        Standalone question:
        """
        prompt_summarize = self.create_prompt_template(
            template=template_summarize, input_variables=["chat_history", "query"]
        )

        llm_summarize = self.create_chat_openai(
            temperature=0, model="gpt-3.5-turbo-1106"
        )

        return LLMChain(llm=llm_summarize, prompt=prompt_summarize)
    
    def answer_question(self, ques):
        if not self.hist:
            response = self.qa_with_source(ques)
            query = response["query"]
            result = response["result"]
            self.hist.append({"query": query, "result": result})
        else:
            hist_lines = self.hist[-10:] if len(self.hist) >= 10 else self.hist[:] 
            standalone_question = self.llm_chain.run(
                {"chat_history": json.dumps(hist_lines), "query": ques}
            )
            print(standalone_question)
            response = self.qa_with_source(standalone_question)
            query = response["query"]
            result = response["result"]
            self.hist.append({"query": query, "result": result})
        return response["result"]
    
    def run_qa_system(self):
        flag = True
        i = 1

        print("\n\n")

        while flag:
            if i == 1:
                question = input("Enter your query. Press 0 to exit.\n")
                if question == "0":
                    flag = False
                    break
                result = self.answer_question(question)
                pprint.pprint(result)

            else:
                question = input("Enter your query. Press 0 to exit.\n")
                if question == "0":
                    flag = False
                    break

                result = self.answer_question(question)
                pprint.pprint(result)

            i += 1

if __name__ == "__main__":
    file_path = '/Users/sarthakgupta/Downloads/floods_in_india.csv'
    floods_qa_system = FloodsQASystem(file_path)
    floods_qa_system.answer_question(ques="When did the last flood occur?")
    floods_qa_system.answer_question(ques="Where did it occur?")
