from elasticsearch import Elasticsearch, exceptions

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200,"scheme": "http"}])

# Define index mapping for storing dense vectors and image paths
index_mapping = {
    "mappings": {
        "properties": {
            "image_vector": {
                "type": "dense_vector",
                "dims": 2048,
                "index": True,
                "similarity": "l2_norm"
            },
            "image_path": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            }
        }
    }
}

# Create the index with the defined mapping
try:
    es.indices.create(index='cbirproject', body=index_mapping)
    print("Index created successfully.")
except exceptions.RequestError as e:
    if e.error == "resource_already_exists_exception":
        print("Index 'cbirproject' already exists. Skipping index creation.")
    else:
        print(f"Failed to create index: {e}")
