
from langchain.document_loaders import CSVLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAI
from langchain.chains import LLMChain
import pprint

loader = CSVLoader(file_path='/Users/sarthakgupta/Downloads/floods_in_india.csv')
docs = loader.load()
print(docs)


OPENAI_API_KEY = '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'

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
context = "\n\n".join(str(p.page_content) for p in docs)
texts = text_splitter.split_text(context)

data = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

store = Chroma.from_documents(
    data, 
    embeddings, 
    ids = [f"{item.metadata['source']}-{index}" for index, item in enumerate(data)],
    collection_name="Floods", 
persist_directory='db',
)
store.persist()

question = " Tell me about Lucknow floods "
docs = vector_index.get_relevant_documents(question)

template = """You are a bot that answers questions about floods, using only the context provided.
If you don't know the answer, simply state that you don't know.

{context}

Question: {question}"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

qa_with_source = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=store.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT, },
    return_source_documents=True,

)


pprint.pprint(
    qa_with_source("When did the last flood occur")
)

pprint.pprint(
    qa_with_source("Compare between 1923 Lucknow Flood and 1971 Lucknow flood")
)

flag = True
i=0

template_summarize = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone 
question.The question should contain as much information as possible about the context.

Chat History:
{chat_history}
Follow Up Input: {query}
Standalone question:
"""
prompt_summarize = PromptTemplate(
    template=template_summarize, input_variables=["chat_history", "query"]
)

llm_summarize=ChatOpenAI(temperature=0,model="gpt-3.5-turbo-1106")
llm_chain=LLMChain(llm=llm_summarize,prompt=prompt_summarize)


hist = ""
flag = True
i=1

print("\n\n")

while(flag==True):
    if(i==1):
        question=input("Enter your query\n")
        if(question=="0"):
            flag = False
            break
        response=qa_with_source(question)
        pprint.pprint(response['result'])
        query = response['query']
        result = response['result']
        hist += f"Query:{query}, Result: {result}\n\n"
        
    else:
        question=input("Enter your query. Press 0 to exit.\n")
        if(question=="0"):
            flag = False
            break
        
        if i>5:
            hist_lines = hist.strip().split("\n\n")
            hist_lines = hist_lines[1:]
            hist = "\n\n".join(hist_lines)
            
        standalone_question=llm_chain.run({"chat_history":hist,"query":question})
        response=qa_with_source(standalone_question)
        print(standalone_question)
        pprint.pprint(response['result'])
        query = response['query']
        result = response['result']
        hist += f"Query:{query}, Result: {result}\n\n"
        
    i+=1


