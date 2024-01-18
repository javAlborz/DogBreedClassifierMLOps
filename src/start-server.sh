#!/bin/bash
python src/download_model_file.py  
exec uvicorn src.api:app --host 0.0.0.0 --port 80