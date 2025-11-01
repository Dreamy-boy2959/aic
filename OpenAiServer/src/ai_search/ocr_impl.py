import json
import logging
from pathlib import Path
from typing import List
import unicodedata
import os

class OCR_Searcher:
    json_dir: Path;
    json_objects = []; 
    
    def __init__(self, json_dir):
        self.json_dir = Path(json_dir);
        self.loadJsonMatrix();
        logging.info("[+] JSON loaded successfully");
        
        
    def loadJsonMatrix(self):
        listDir = os.listdir(self.json_dir);
                
        for vidDir in listDir:
            if vidDir != '.gitkeep':
                all_json_data = []
                files = [os.path.join(self.json_dir, vidDir, x) for x in os.listdir(os.path.join(self.json_dir, vidDir))]; # Guranteed to be all files, no need to filter
                
                logging.info(f"Directory {vidDir} has {len(files)} files");
                
                for file in files:
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            all_json_data.append(data)
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON from {file}: {e}")
                    except FileNotFoundError:
                        logging.error(f"File not found: {file}")
                    
        self.json_objects.append(all_json_data);
        
    def compare(self, queryStr : str, limit : int) -> List[str]:
        if len(self.json_objects) == 0:
            logging.error("[-] Still initializing, please try again")
            return
        
        result = [];
        for vidDir in self.json_objects:
            for file in vidDir:
                for entry in file:
                    # making results consistent by removing diacritic
                    processed_list = list(set([self.remove_diacritics(x).lower() for x in entry['Txt']]))
                    for string in processed_list:                
                        if self.remove_diacritics(queryStr) in string:  
                            result.append((entry['Vid_id'][:3] + '/' + entry['Vid_id'] + '/' + entry['Keyframe_id']));
                    
        logging.info(f"[+] OCR hit for query {queryStr}, found {len(result)} results");
        return result
    
    def remove_diacritics(self, text : str):
        normalized_text = unicodedata.normalize('NFKD', text)
        return "".join([c for c in normalized_text if not unicodedata.combining(c)])