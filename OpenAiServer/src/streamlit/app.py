import streamlit as st
import requests
import json
import time
import io
from PIL import Image
from streamlit_paste_button import paste_image_button

# --- Configuration ---
st.set_page_config(page_title="AI Challenge Search", layout="wide")

# Backend server configuration
# Assumes your server has POST endpoints
BASE_URL = "http://localhost:5000"
CLIP_ENDPOINT = f"{BASE_URL}/search/text"
OCR_ENDPOINT = f"{BASE_URL}/search/ocr"
ASR_ENDPOINT = f"{BASE_URL}/search/asr"
IMAGE_SEARCH_ENDPOINT = f"{BASE_URL}/search/image" # New endpoint for image search

# --- Helper Functions ---
@st.cache_data(ttl=30) # Cache for 30 seconds
def check_endpoint_status(url):
    """
    Checks if an endpoint is reachable.
    Returns (True, "Reachable") or (False, "Unreachable").
    We assume *any* response (even 405, 500) means it's reachable.
    Only connection errors or timeouts mean it's unreachable.
    """
    try:
        # Use HEAD request for a lightweight check.
        response = requests.head(url, timeout=3)
        return True, "Reachable"
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False, "Unreachable"
    except requests.exceptions.RequestException as e:
        # If HEAD fails (e.g., 405 Method Not Allowed), try GET
        try:
            response = requests.get(url, timeout=3)
            return True, "Reachable" # Any response code is fine
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False, "Unreachable"
        except Exception as get_e:
            # Catch any other GET errors
            return False, f"Unreachable ({get_e})"
        
def fetch_results_with_retry(endpoint_url, query, retries=3, delay=2, limit=200):
    """
    Fetches search results from a given API endpoint with exponential backoff.
    
    Args:
        endpoint_url (str): The URL of the API endpoint.
        query (str): The search query text.
        retries (int): Number of retries.
        delay (int): Initial delay in seconds.
        
    Returns:
        A tuple (data, error_message). 
        'data' is the JSON response if successful, else None.
        'error_message' is a string if an error occurred, else None.
    """
    payload = {"query": query, "limit": limit}
    headers = {"Content-Type": "application/json"}
    
    for i in range(retries):
        try:
            response = requests.get(
                endpoint_url, 
                params=payload, 
                headers=headers, 
                timeout=30 # 30-second timeout
            )
            
            # Check for successful response
            if response.status_code == 200:
                try:
                    return response.json(), None  # (data, error)
                except json.JSONDecodeError:
                    return None, "Error: Failed to decode JSON response from server."
            else:
                return None, f"Error: Server returned status code {response.status_code}. Response: {response.text}"
        
        except requests.exceptions.ConnectionError:
            error_msg = f"Error: Could not connect to the server at {endpoint_url}. Is the server running?"
            # This is a connection error, so we retry
        except requests.exceptions.Timeout:
            error_msg = "Error: The request timed out."
            # Timeouts can be retried
        except Exception as e:
            # Other unexpected errors
            return None, f"An unexpected error occurred: {e}"
        
        # If we are here, it means a retryable error occurred
        if i < retries - 1:
            # Don't log to console, just wait and retry
            time.sleep(delay)
            delay *= 2 # Exponential backoff
        else:
            # This was the last retry
            return None, error_msg
            
    return None, "Error: Unknown error after all retries."

def fetch_image_search_results(endpoint_url, file_obj, file_name, file_type, retries=3, delay=2):
    """
    Fetches search results by uploading an image to the API endpoint.
    
    Args:
        endpoint_url (str): The URL of the API endpoint.
        file_obj (file-like): The file-like object (e.g., UploadedFile or io.BytesIO).
        file_name (str): The filename to send to the server.
        file_type (str): The MIME type of the file (e.g., "image/png").
        retries (int): Number of retries.
        delay (int): Initial delay in seconds.
        
    Returns:
        A tuple (data, error_message).
    """
    for i in range(retries):
        try:
            # IMPORTANT: Reset file pointer for each attempt (in case of retries)
            file_obj.seek(0)
            
            # Prepare the file for multipart/form-data upload
            # The server should expect a file field named 'image_file'
            files = {'image_file': (file_name, file_obj, file_type)}
            
            response = requests.post(
                endpoint_url, 
                files=files, 
                timeout=30 # 30-second timeout
            )
            
                # Print request details
            print(f"Request URL: {response.request.url}")
            print(f"Request Headers: {response.request.headers}")
            print(f"Request Body: {response.request.body[:100]}...") # For POST/PUT requests
            
            # Check for successful response
            if response.status_code == 200:
                try:
                    return response.json(), None  # (data, error)
                except json.JSONDecodeError:
                    return None, "Error: Failed to decode JSON response from server."
            else:
                return None, f"Error: Server returned status code {response.status_code}. Response: {response.text}"
        
        except requests.exceptions.ConnectionError:
            error_msg = f"Error: Could not connect to the server at {endpoint_url}. Is the server running?"
        except requests.exceptions.Timeout:
            error_msg = "Error: The request timed out."
        except Exception as e:
            return None, f"An unexpected error occurred: {e}"
        
        # If we are here, it means a retryable error occurred
        if i < retries - 1:
            time.sleep(delay)
            delay *= 2
        else:
            # This was the last retry
            return None, error_msg
            
    return None, "Error: Unknown error after all retries."


def display_results(results):
    """
    Displays the search results in a responsive grid.
    
    Args:
        results (list): A list of result objects, where each object
                        is expected to have 'image' and 'resourceId'.
    """
    
    st.markdown(f"#### Faiss Results - {len(results['faiss'])} entries")
    
    if not results:
        st.info("No results found for your query.")
        return
            
    # Create a responsive grid. Adjust '4' for more/fewer columns on desktop.
    cols = st.columns(4)
    for i, item in enumerate(results['faiss']):
        col = cols[i % 4]
        with col:
            try:
                image_url = item.get("url").replace(".jpg", ".webp")
                id = item.get("id") if item.get("id") is not None else 'Not Found'
                
                if not id:
                    st.warning("Result item is missing 'resourceId'. Skipping.")
                    continue

                # Display image if URL is provided
                if image_url:
                    st.image(
                        image_url, 
                        caption=f"ID: {id}", 
                        width="stretch",
                    )
                else:
                    # Fallback if no image URL is provided
                    st.error(f"Image not available for ID: {id}")
                
                st.write(f"**Resource ID:**")
                # Make resource ID copyable
                st.code(id, language=None)
                st.divider()

            except Exception as e:
                st.error(f"Error displaying item: {e}")

# --- Main Application UI ---

st.title("🏆 AI Challenge Search Interface")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Text Search", 
    "Image Search"
])

st.sidebar.markdown("---")

# --- Status Widget ---
st.sidebar.subheader("Endpoint Status")
if page == "Text Search":
    # For this tab, we'll just ping the CLIP endpoint as a representative
    endpoint_to_check = CLIP_ENDPOINT
    label = "Text Search (CLIP)"
else: # page == "Image Search"
    endpoint_to_check = IMAGE_SEARCH_ENDPOINT
    label = "Image Search"
    
status, message = check_endpoint_status(endpoint_to_check)

if status:
    st.sidebar.success(f"{label}: {message}")
else:
    st.sidebar.error(f"{label}: {message}")
st.sidebar.caption(f"Pinging: `{endpoint_to_check}`")

st.sidebar.markdown("---")
st.sidebar.info(f"This app connects to a backend server expected to be running at `{BASE_URL}`.")

# --- Page 1: Text Search (Combined) ---
if page == "Text Search":
    st.header("🔍 Text Search")
    
    search_type = st.selectbox(
        "Select search type:",
        ["Plain Search (CLIP)", "OCR Search", "ASR Search"]
    )
    
    # Configuration for each search mode
    SEARCH_MODES = {
        "Plain Search (CLIP)": {
            "endpoint": CLIP_ENDPOINT,
            "prompt": "Search for images using natural language descriptions (e.g., 'a photo of a dog playing in a park').",
            "label": "Enter your search query:",
            "spinner": "Searching with CLIP model..."
        },
        "OCR Search": {
            "endpoint": OCR_ENDPOINT,
            "prompt": "Search for images that contain specific text (e.g., 'DANGER', 'Main Street', 'Analytics').",
            "label": "Enter text to find in images:",
            "spinner": "Searching OCR data..."
        },
        "ASR Search": {
            "endpoint": ASR_ENDPOINT,
            "prompt": "Search for video frames based on spoken words from subtitles (e.g., 'welcome back to the channel', 'financial report').",
            "label": "Enter spoken words to find:",
            "spinner": "Searching ASR data..."
        }
    }
    
    mode_config = SEARCH_MODES[search_type]
    
    st.markdown(mode_config["prompt"])
    
    with st.form(key="text_search_form"):
        query = st.text_input(mode_config["label"], key="text_query")
        submit_button = st.form_submit_button(label="Search")
    
    if submit_button and query:
        with st.spinner(mode_config["spinner"]):
            results, error = fetch_results_with_retry(mode_config["endpoint"], query)
            
            if error:
                st.error(error)
            elif results is not None:
                display_results(results)

# --- Page 2: Image Search ---
elif page == "Image Search":
    st.header("🖼️ Image Search")
    st.markdown("Search for similar images by uploading an image.")
    
    with st.form(key="image_search_form"):
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "png", "jpeg", 'webp'], 
            accept_multiple_files=False
        )
        submit_button = st.form_submit_button(label="Search")
        
    if submit_button and uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Your uploaded image", width=300)
        
        # Search
        with st.spinner("Searching for similar images..."):
            results, error = fetch_image_search_results(IMAGE_SEARCH_ENDPOINT, uploaded_file, uploaded_file.file_id, uploaded_file.type)
            
            if error:
                st.error(error)
            elif results is not None:
                display_results(results)
                
    st.markdown("---")
    st.write("Or paste an image:")

    # 2. Paste Option
    paste_btn = paste_image_button(
        label="📋 Paste from Clipboard",
        key="paste_btn",
        background_color="#FF4B4B",
        hover_background_color="#FF6B6B"
    )
    
    if paste_btn.image_data is not None:
        # Display the pasted image
        pil_image = paste_btn.image_data
        st.image(pil_image, caption="Your pasted image", width=300)
        
        # Search button for pasted image
        if st.button("Search with Pasted Image", key="search_paste"):
            # Convert PIL Image to file-like object (io.BytesIO)
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            buf.seek(0)
            
            with st.spinner("Searching for similar images..."):
                results, error = fetch_image_search_results(
                    IMAGE_SEARCH_ENDPOINT,
                    buf,
                    "pasted_image.png",
                    "image/png"
                ) 
                
                if error:
                    st.error(error)
                elif results is not None:
                    display_results(results)
                
    elif submit_button and uploaded_file is None:
        st.warning("Please upload an image first.")