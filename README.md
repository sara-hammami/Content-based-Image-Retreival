# Content-Based-Image-Retreival

Welcome to our Content-Based Image Retrieval (CBIR) project! This project is aimed at implementing a simple content-based image retrieval system using feature retrieval with ResNet50 as the backbone. The system is backed by Elasticsearch, providing efficient and powerful searching capabilities. Additionally, it offers text-based search functionality. Below are the instructions to set up and run the CBIR system.

## Project Overview

The CBIR system consists of the following components:
- **Feature Extraction:** Utilizes ResNet50 for feature extraction from images.
- **Elasticsearch:** Provides indexing and searching capabilities for images and text data.
- **Local Dataset:** A Flickr dataset, which should be placed under the `dataset` directory.
- **Text-Based Search:** Based on URLs, allowing text-based search functionality.

## Setting Up the System

1. **Install Dependencies:**
   - Ensure Python is installed on your system.
   - Install required Python libraries using `pip install -r requirements.txt`.
   - install logstash  version 8.10.4

2. **tag based indexing :**
   - Start Elasticsearch on your system.
   - Create an index by running `curl -X PUT "http://localhost:9200/flickerphotos" -H 'Content-Type: application/json' -d @mapping.json'
   - Push the data to Elasticsearch using Logstash to populate the created index.Run this command : logstash.bat -f phtotos_flickr_conf_73.conf

3. **Content-Based Image Retrieval Setup:**
   - Run `mapping.py` to create the index for content-based image retrieval.
   - Use ResNet50 for feature extraction and store the features in the Elasticsearch index.
   -  The indexing.py script is responsible for indexing images.
   - The dataset used is not  in this repo but it should be palaced here (locally ).

## Running the System
- Run the CBIR system by executing the app.py script.
- Use the provided search functionalities to perform content-based and text-based image searches.




