import os
from pathlib import Path
import faiss
import json
import torch
import open_clip
from PIL import Image
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from io import BytesIO
import numpy as np

class FaissSearch:
    def __init__(self, index_dir: str, metadata_dir: str, device: str = "cuda"):
        logging.info(f"[+] Initializing FaissSearch with device: {device}")
        
        index_path = Path(index_dir)
        self.metadata_dir = Path(metadata_dir)
        self.device = device
        self.indices: List[Tuple[Path, faiss.Index]] = [] 

        logging.info("[+] Loading index files...")
        # Use glob to find all .faiss files
        index_files = list(index_path.glob('*.faiss'))
        
        if not index_files:
            logging.warning(f"[-] No .faiss files found in {index_dir}")
            
        for file_path in index_files:
            try:
                index = faiss.read_index(str(file_path))
                self.indices.append((file_path, index))
            except Exception as e:
                logging.error(f"[-] Failed to load index {file_path}: {e}")
        
        logging.info(f"[+] {len(self.indices)} index files loaded successfully")
                
        # Load CLIP model
        logging.info("[+] Loading CLIP model...")
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
                )
            self.model = self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode
            self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise # Fatal for this routine

        assert self.model is not None
        assert self.tokenizer is not None
        assert self.device is not None
        logging.info("[+] FaissSearch initialized successfully.")

    def searchImage(self, binaryData, limit: int) -> List[str]:
        logging.info("[+] Searching image");
        
        img_tensor = self.preprocess(Image.open(binaryData).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.amp.autocast(device_type=self.device):
            img_features = self.model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            image_vector = img_features.cpu().numpy().astype("float32")
            image_vector = np.sign(image_vector) * np.power(np.abs(image_vector), 0.6)
            image_vector /= np.linalg.norm(image_vector, axis=1, keepdims=True)
           
        return self.mapToResult(image_vector, limit)     
            
            
    def searchText(self, searchType, query: str, limit: int) -> List[str]:
        logging.info(f"[+] Encoding query: '{query[:50]}...'")
        try:
            with torch.no_grad(), torch.amp.autocast(device_type=self.device):
                text_tokens = self.tokenizer([query]).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Faiss requires a float32 numpy array
                text_vector = text_features.cpu().float().numpy().astype('float32')
        except Exception as e:
            logging.error(f"Failed to encode query: {e}")
            return []
        
        return self.mapToResult(text_vector, limit)

    def mapToResult(self, text_vector, limit): 
        # This list will store tuples of (distance, img_path)
        all_results: List[Tuple[float, str]] = [] 

        with ThreadPoolExecutor() as executor:
            logging.info(f'[+] Spawning {len(self.indices)} search threads...')
    
            futures_map = {
                executor.submit(index_content.search, text_vector, limit): index_path
                for index_path, index_content in self.indices
            }

            logging.info('[+] Waiting for search results...')
            for future in as_completed(futures_map):
                index_path = futures_map[future]
                
                try:
                    distances, indices = future.result()
                except Exception as e:
                    logging.error(f"Search failed for index {index_path.name}: {e}")
                    continue

                metadata_name = index_path.stem + ".json"
                metadata_file_path = self.metadata_dir / metadata_name
                
                if not metadata_file_path.is_file():
                    logging.warning(f"Metadata file not found, skipping: {metadata_file_path}")
                    continue
                
                try:
                    # Load metadata only once per index
                    with open(metadata_file_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logging.error(f"Failed to load metadata {metadata_file_path}: {e}")
                    continue # Skip this index's results

                # logging.info(f"[+] Mapping results from {index_path.name}")
                
                for dist, idx in zip(distances[0], indices[0]):
                    if idx == -1:
                        continue
                    
                    try:
                        # JSON keys are strings, so convert index
                        img_path = metadata[str(idx)]
                        all_results.append((float(dist), img_path))
                    except KeyError:
                        logging.warning(f"Index {idx} not found in metadata {metadata_file_path.name}")
                    except Exception as e:
                        logging.error(f"Error processing metadata for idx {idx}: {e}")

        # Get global Top-K results
        logging.info(f"[+] Found {len(all_results)} total results. Sorting for top {limit}...")
        # all_results.sort(key=lambda x: x[0])
        final_results = [img_path for dist, img_path in all_results[:limit]]
        
        logging.info(f"[+] Returning {len(final_results)} results.")

        if len(final_results) < 12:
             logging.info(f"[+] Results: {final_results}")
             
             
        print(final_results)
        return final_results