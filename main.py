from sentence_transformers import SentenceTransformer
import config
from qdrant_client import QdrantClient
from qdrant_client.http.models import NamedVector
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, flash, redirect
from PIL import Image
from werkzeug.utils import secure_filename
import io
import os
import base64

app = Flask(__name__, template_folder='./Template')
app.config['SECRET_KEY'] = 'very_s'

#connect to the client
client_q = QdrantClient( url=config.qdrant_url, api_key=config.qdrant_api_key )

#get the model
model_image = SentenceTransformer("sentence-transformers/clip-ViT-B-32",
                                  cache_folder="./model")
# get the data
dataset = load_dataset("fashion_dataset", split="train")

#create index
values= list(range(len(dataset)))
dataset=dataset.add_column('image_id',values)

#for debug purpose
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#name your collection
my_collection = "image_collection_test_2"

#a folder directory for uploading functions
app.config['UPLOAD_FOLDER'] = 'uploads'

#allowed extensions for image data
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def search_qdrant(data, name='productDisplayName', query_filter = None, limit=10):
  results = client_q.search(
    collection_name=my_collection,
    query_vector=(name, model_image.encode(data).tolist()),
    limit=limit,
    with_vectors=True,
    query_filter=query_filter,
    with_payload=True,)
  id_list = [point.id for point in results]
  Score_list = [point.score for point in results]
  return dict(zip(id_list, Score_list))



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    image_responses = []
    if request.method == 'POST':
        # Check if the post request has the file part
        if request.files:
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)

            file = request.files['file']

            if file and allowed_file(file.filename):
                # Save the uploaded image
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Read the image with PIL
                uploaded_image = Image.open(file_path)

                # Perform the search
                id_list = search_qdrant(uploaded_image, name='image', limit=10)

                # Filter the fashion_dataset based on the search results
                filtered_dataset = dataset.select(id_list.keys())

                # Display the images

                for example in filtered_dataset:
                    image_bytes = example["image"]

                    if isinstance(image_bytes, bytes):
                        # If the image is in bytes, convert it to a PIL image
                        image_bytes = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                    buffered = io.BytesIO()
                    image_bytes.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_responses.append(img_str)

            elif request.form['text_input']:
                print(request.form['text_input'])
                # Perform the search
                id_list = search_qdrant(request.form['text_input'], name='productDisplayName', limit=10)
                print(id_list)
                # Filter the fashion_dataset based on the search results
                filtered_dataset = dataset.select(id_list.keys())

                # Display the images
                for example in filtered_dataset:
                    image_bytes = example["image"]

                    if isinstance(image_bytes, bytes):
                        # If the image is in bytes, convert it to a PIL image
                        image_bytes = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                    buffered = io.BytesIO()
                    image_bytes.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_responses.append(img_str)
    return render_template('index.html', images=image_responses)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    print(f"Port: {port}")
    app.run(host='0.0.0.0', port=port)

""" 
step 0: build
docker buildx build --platform linux/amd64 -t searchapp .
step 1: tag
docker tag searchapp gcr.io/project-name/searchapp
step 2: push
docker push gcr.io/project-name/searchapp 
step 3: run
docker run -p 80:80 --platform linux/amd64 -d searchapp
docker run -d -p 80:80 searchapp
"""
