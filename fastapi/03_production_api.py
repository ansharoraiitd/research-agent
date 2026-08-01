#Section 1 - imports 
import sys 
import os 
import time 
import json 
import uuid 
import asyncio 
import logging 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request 
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import StreamingResponse, JSONResponse 
from pydantic import BaseModel 
from typing import Optional, AsyncGenerator 
import uvicorn 
from dotenv import load_dotenv 

load_dotenv()

#logging instead of print() - professional standard 