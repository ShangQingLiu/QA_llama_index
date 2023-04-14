from flask import Flask,request,jsonify
from config import Config
from utils import IndexUtils

import os

app = Flask(__name__)
app.config.from_object(Config)
os.environ['OPENAI_API_KEY'] = app.config['OPENAI_API_KEY'] 


@app.route("/")
def hello_world():
    return "<p>This is the QA system based on llama index!</p>"



# Upload File
@app.route("/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'project_name' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    project_name = request.files['project_name']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    upload_folder = os.join(app.config['UPLOAD_FOLDER'], project_name)
    # Check if the upload directory exists, create it if not
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Save the file to the UPLOAD_FOLDER
    file.save(os.path.join(upload_folder, file.filename))

    return jsonify({'message': 'File uploaded successfully'}), 200

# Create Index 
@app.route("/createIndex", methods=['POST'])
def create_index():
    if 'project_name' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    project_name = request.files['project_name']
    IndexUtils(app.config["INDEX_SAVE_PATH"],project_name).buildIndexer()




# Utils
## DataLoader(Audio, Docx, PDF, Html)
## SaveIndexer(chunk_size_limit,data)
## LoadIndexer(path) 
## SaveGraphIndexer(Indexers)
## LoadGraphIndexer(path)

# Chatbot
## Init(Index_sets)
## run(query) -> return answer