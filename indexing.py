import torch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from src.feature_extraction import MyResnet50
from src.dataloader import MyDataLoader

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200,"scheme": "http"}])

# Initialize MyResnet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
restnet_model = MyResnet50(device=device)

# Initialize MyDataLoader
image_root = 'D:/AIM-tbourbi/projetCBIR/training_set'
data_loader = MyDataLoader(image_root)
#print(len(data_loader))

# Prepare actions for bulk indexing  
actions = []
for idx in range(len(data_loader)):
    image, image_path = data_loader[idx]
    image = image.unsqueeze(0).to(device)
    feature = restnet_model.extract_features(image)
    image_path_str = str(image_path).replace("\\", "/").strip('/')

    # Prepare document for Elasticsearch
    doc = {
        '_op_type': 'index',
        '_index': 'cbirproject',
        '_id': idx,
        '_source': {
            'image_path': image_path_str,
            'image_vector': feature
        }
    }
    actions.append(doc)

# Bulk index documents
try:
    success, _ = bulk(es, actions=actions, raise_on_error=True)
    print(f"Successfully indexed {success} documents.")
except Exception as e:
    print(f"Failed to index documents: {e}")
