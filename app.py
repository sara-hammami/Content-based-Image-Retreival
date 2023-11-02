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
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            #Load, transform, and extract features from the uploaded image
            image = Image.open(filename)
            transform=get_transformation()
            image_tensor=transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            feature = resnet_model.extract_features(image_tensor)

           # Search for similar_images in Elasticsearch
            similar_images = search_similar_images(feature)
            return render_template('index.html', image_path=filename, similar_images=similar_images)

    return render_template('index.html', error=None, image_path=None, similar_images=None ) ,{'Content-Language': 'en'}
@app.route('/images/<path:image_path>')
def serve_image(image_path):
    return send_file(image_path, mimetype='image/jpeg')# Adjust mimetype if necessary

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        # Check if the search query is in the form data
        if 'search_query' in request.form:
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
            # Render the template with text search results
            return render_template('index.html', res=res)

    # Handle other cases (GET request, no search query, etc.)
    return render_template('index.html')
if __name__ == '__main__':
<<<<<<< Updated upstream
    app.run(debug=True)

=======
    app.run(debug=True)
>>>>>>> Stashed changes
