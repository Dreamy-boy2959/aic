import torch
import logging

from .faiss_impl import FaissSearch
import os
import torch

class FaissSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, index_dir, metadata_dir):
        logging.info("[+] Initializing FaissSingleton...")
        # Initialization logic here (called only once for the first instance)
        self._instance = FaissSearch(
            index_dir,
            metadata_dir,
            "cuda" if torch.cuda.is_available() else "cpu")
        logging.info("[+] Initializing completed")
        torch.cuda.empty_cache()
        pass

    def get_instance(self):
        return self._instance