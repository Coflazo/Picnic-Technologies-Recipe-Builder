#!/usr/bin/env python3

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import uvicorn

def main():
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False, 
        log_level="info",
    )

if __name__ == "__main__":
    main()