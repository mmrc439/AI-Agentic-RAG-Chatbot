from dotenv import load_dotenv
import os

# Load .env file into environment
load_dotenv()

DEBUG = int(os.getenv("DEBUG", 0))