import os
import logging
import json

from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
from pathlib import Path

from ai_search.faiss_singleton import FaissSingleton
from ai_search.ocr_impl import OCR_Searcher

from filters.ExcludeFileFilter import ExcludeFileFilter

# Configure logging before creating the app
streamHandler = logging.StreamHandler();
streamHandler.addFilter(ExcludeFileFilter(['open_clip']))
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[streamHandler, logging.FileHandler("logs/latest.txt")])

app = Flask(__name__)
CORS(app) # Enables CORS for all routes and origins by default

# Build absolute path to your images folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Root/
IMAGE_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "res", "images", "keyframes"))
OCR_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "res", "ocrs"))
INDEX_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "res", "indices"))
METADATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "res", "metadatas"))
VIDEO_TO_ID_FILE = os.path.normpath(os.path.join(BASE_DIR, "..", "res", "videoToId.json"))

# Asserst the path exist
assert Path(IMAGE_DIR).is_dir() == Path(OCR_DIR).is_dir() == Path(INDEX_DIR).is_dir() == Path(METADATA_DIR).is_dir() == Path(VIDEO_TO_ID_FILE).is_file();
assert len(os.listdir(METADATA_DIR)) == len(os.listdir(INDEX_DIR))
app.logger.info("[+] All resources present, ready to go!")

app.logger.info("================ FILE/DIR PATH ====================")
app.logger.info(f"[+] Base dir: {BASE_DIR}")
app.logger.info(f"[+] Video to ID json: {VIDEO_TO_ID_FILE}")
app.logger.info(f"[+] Image dir: {IMAGE_DIR} - Found {len(os.listdir(IMAGE_DIR))} image directories")
app.logger.info(f"[+] OCR dir: {OCR_DIR} - Found {len(os.listdir(OCR_DIR))} ocr directories")
app.logger.info(f"[+] Index dir: {INDEX_DIR} - Found {len(os.listdir(INDEX_DIR))} index files")
app.logger.info(f"[+] Metadata dir: {METADATA_DIR} Found {len(os.listdir(METADATA_DIR))} metadata files")
app.logger.info("===================================================")

TEMP_ENDPOINT = "http://localhost:5000"

# INITIALIZATION 
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    # Only runs in the *real* reloader process
    faiss_instance = FaissSingleton(INDEX_DIR, METADATA_DIR).get_instance()
    ocr_instance = OCR_Searcher(OCR_DIR);


@app.route("/images/<path:filename>")
def serve_image(filename : str):
    return send_from_directory(IMAGE_DIR, filename)

@app.route("/search/image", methods=['POST'])
def search_image():
    if request.method == "POST":
        image_payload = request.files.get("image_file").stream
        faiss_result = faiss_instance.searchImage(image_payload, request.args.get('limit') if request.args.get('limit') is not None else 10);

        with open(VIDEO_TO_ID_FILE, 'r', encoding='utf-8') as file:
            data = json.load(file);
            
            faiss_result = [{
                "url": f"{TEMP_ENDPOINT}/images/{result}",
                "id": data.get(result.split("/")[1])
            } for result in faiss_result]
        
        return {
            "faiss": faiss_result
        }
    else:
        logging.error("[-] Wrong method for image searcher");
    

@app.route('/search/text')
def search_text():
    search_type = request.args.get('searchType', '', type=str)
    query = request.args.get('q', '', type=str)
    limit = request.args.get('limit')

    if len(query) == len(limit) == 0:
        logging.error("[-] Error while parsing request queries");
        return;
    
    limit = int(limit);

    # Log it out
    faiss_result = faiss_instance.searchText(search_type, query, limit)
    ocr_result = ocr_instance.compare(query, limit)

    results = set(faiss_result).intersection(set(ocr_result));
    # print(faiss_result, ocr_result)
    logging.info(f"[+] Found {len(results)} for query {query}");

    with open(VIDEO_TO_ID_FILE, 'r', encoding='utf-8') as file:
        data = json.load(file);
        
        faiss_result = [{
            "url": f"{TEMP_ENDPOINT}/images/{result}",
            "id": data.get(result.split("/")[1])
        } for result in faiss_result]
        
        ocr_result = [{
            "url": f"{TEMP_ENDPOINT}/images/{result}",
            "id": data.get(result.split("/")[1])
        } for result in ocr_result][:limit]
        
        intersection_result = [{
            "url": f"{TEMP_ENDPOINT}/images/{result}",
            "id": data.get(result.split("/")[1])
        } for result in list(results)]


    return {
        "faiss": faiss_result,
        "ocr": ocr_result,
        "intersection": intersection_result
    }
    
@app.route('/mlt')
def moreLikeThis():
    vidID = request.args.get('vidID').replace("/", os.sep)
    fullPath = os.path.join(IMAGE_DIR, vidID)
    RETURN_THRESHOLD = 8;

    if os.path.exists(fullPath):
        fileName = os.path.basename(fullPath)
        allFileFromParent = os.listdir(os.path.dirname(fullPath))
        allFileFromParent.sort(reverse=True)
        targetIdx = allFileFromParent.index(fileName)
        trimmedList = allFileFromParent[targetIdx - RETURN_THRESHOLD:targetIdx + RETURN_THRESHOLD]
        return {
            "results": [f"{TEMP_ENDPOINT}/images/{vidID.split(os.sep)[0]}/{vidID.split(os.sep)[1]}/{x}" for x in trimmedList]
        }

    return {
        "results": []
    }

@app.route('/health')
def health():
    return {
        "status": "ok",
    }


if __name__ == '__main__':
    app.run(debug=False)
