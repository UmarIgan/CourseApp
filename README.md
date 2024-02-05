# Introduction to Vector Databases and Multi-Modal Semantic Search

Thanks for takin this course, The codes we showed on the course can be seen here.
To run application please follow those steps:

## Introduction

Vector databases are indispensable for applications requiring similarity search, including recommendation systems, content-based image retrieval, and personalized search. Leveraging efficient indexing and searching techniques, vector databases enhance the speed and accuracy of retrieving unstructured data already represented as vectors.

## Embeddings

Embedding models play a crucial role in semantic search, allowing us to represent text, image, audio, or video data numerically. These models, whether sparse or dense, compress information into high-dimensional vectors. We utilize the CLIP (Contrastive Language-Image Pretraining) model for multimodal search.

## CLIP Model

Developed by OpenAI in 2021, CLIP is a Contrastive Language-Image Pretraining model trained on over 400 million images and their text representations. Its encoder handles both image and text data, making it efficient and lightweight (only 600 MB). While used for search operations in this demonstration, CLIP can also classify images similar to other pre-trained models like ResNet.

## Dataset

The dataset used in this project is sourced from a huggingface image dataset. This dataset comprises images of fashion products along with their titles, color, size, etc. totaling approximately 44,000 rows.
Dataset's text were embedded using sentence-transformers/all-MiniLM-L6-v1 model while images embedded using CLIP model.

Feel free to explore the provided Jupyter notebook to delve into the implementation details of multimodal search using any vector database and CLIP embeddings.

## Dependencies

To run this project, follow these steps:

1. **Clone The Application:**

   ```bash
   git clone https://github.com/UmarIgan/CourseApp.git
   ```

2. **Create Virtual Environment:**

   ```bash
   python -m venv venv
   ```

3. **Install Dependencies:**

   ```bash
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   pip install -r requirements.txt
   ```

4. **Run the Application:**

   ```bash
   python main.py
   ```
5. **Wait for Model Uploads:**

   - After activating the virtual environment, wait for the image and text models to upload. This may take a few minutes.

6. **Wait for Dataset Download:**

   - Ensure that the required dataset is downloaded. This process will also take some time.
7. **Access the Application:**

   - Once the application is running, navigate to the provided localhost URL (usually http://127.0.0.1:5000/) in your web browser.

   - The application page will allow you to search for images by text or image. For text searches, the `sentence-transformers/all-MiniLM-L6-v1` model is used, and for image searches, the CLIP model is employed.

Feel free to explore the provided Jupyter notebook to delve into the implementation details of multimodal search using various vector databases and CLIP embeddings.

