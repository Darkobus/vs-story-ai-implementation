import chromadb

client = chromadb.PersistentClient(path="./wiki_db")
collections = client.list_collections()

for col in collections:
    print(col.name)
