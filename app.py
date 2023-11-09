from flask import Flask, render_template, request
from elasticsearch import Elasticsearch
from scipy.spatial import distance
from src.feature_extraction import MyResnet50
from PIL import Image
import torch
from torchvision import transforms
from src.dataloader import get_transformation
import os 
from flask import send_file
import requests
app = Flask(__name__)
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = MyResnet50(device=device)



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg' , 'png'}

def search_similar_images(feature_vector, index_name='cbirproject'):
    # Construct Elasticsearch query to calculate Euclidean distance and sort by it
    search_body = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "1 / (1 + l2norm(params.query_vector, 'image_vector'))",
                    "params": {
                     "query_vector": feature_vector
                    }
                }
            }
        }
    }
    
    # Execute the Elasticsearch search query
    #print("Elasticsearch Query Body:", search_body)
    response = es.search(index=index_name, body=search_body)

    # Extract similar images from the search results
    
    similar_images = []
    for hit in response['hits']['hits']:
        similar_images.append(hit['_source']['image_path'])

    return similar_images




@app.route('/', methods=['GET', 'POST'])
def index():
    similar_images = None
    text_search_results = None
    

    if request.method == 'POST':
        if 'file' in request.files:
            # Handle image upload logic
            file = request.files['file']
            if file and allowed_file(file.filename):
                # Save the uploaded file
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)

                # Load, transform, and extract features from the uploaded image
                image = Image.open(filename)
                transform = get_transformation()
                image_tensor = transform(image)
                image_tensor = image_tensor.unsqueeze(0).to(device)
                feature = resnet_model.extract_features(image_tensor)

                # Search for similar_images in Elasticsearch
                similar_images = search_similar_images(feature)
        elif 'search_query' in request.form:
            # Handle text search logic
            search_term = request.form['search_query']
            # Execute Elasticsearch search query for text search
            res = es.search(
                index="flickerphotos",
                size=30,
                body={
                    "query": {
                        "multi_match": {
                            "query": search_term,
                            "fields": ["url", "title", "tags"],
                            "fuzziness": "AUTO",
                            "prefix_length": 1
                        }
                    }
                }
            )
            
            # Extract text search results
            text_search_results = res['hits']['hits']
            

    return render_template('index.html', similar_images=similar_images, text_search_results=text_search_results)
@app.route('/images/<path:image_path>')
def serve_image(image_path):
    return send_file(image_path, mimetype='image/jpeg')# Adjust mimetype if necessary
if __name__ == '__main__':
    app.run(debug=True)