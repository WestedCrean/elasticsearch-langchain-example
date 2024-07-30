from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


embeddings_model = OllamaEmbeddings(model="llama3.1")


loader = TextLoader("../lotr_test_file.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)
docs = text_splitter.split_documents(documents)

db = ElasticsearchStore.from_documents(
    docs,
    embeddings_model,
    es_url="http://localhost:9200",
    index_name="test-basic",
    strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True)
)


db.client.indices.refresh(index="test-basic")

query = "Sauron"
results = db.similarity_search(query, k=3)

print(f"-----------------")
print(f"Query: {query}")
print(f"-----------------")
print(f"Length of result: {len(results)}\n")
for i, r in enumerate(results):
    print(f"\t{i+1}. {r.page_content}\n")

# clear the index
db.client.indices.delete(index="test-basic")
print("Deleted index test-basic")