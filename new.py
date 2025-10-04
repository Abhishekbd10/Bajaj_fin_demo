from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# 1. Load PDFs
pdf_files=["Earnings Call Transcript FY26 - Q1.pdf","Bajaj Finserv Investor Presentation - FY2025-26 - Q1.pdf"]
Documents=[]
for i in pdf_files:
    loader=PyPDFLoader(i)
    docs=loader.load()
    Documents.extend(docs)

# 2. Text Splitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=40)
split_docs=text_splitter.split_documents(Documents)

# 3. Embeddings
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
vectordb=FAISS.from_documents(split_docs,embedding=embeddings)

# 4. Retriever in Vector Store
retriever=vectordb.as_retriever(search_kwargs={"k":3})

# 5. Load LLM
llm=ChatOpenAI(model="gpt-5",temperature=0.2)

# 6. Create chain
chain= retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)) | llm

# 7. Query
query=input("Enter your query: ")
response=chain.invoke(query)

print(query)
print(response.content)