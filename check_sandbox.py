"""Check what your Nebius key is allowed to do in Sandboxes."""
import os
from dotenv import load_dotenv
from contree_client.httpx import ContreeClient

load_dotenv()
key, project = os.environ.get("NEBIUS_API_KEY"), os.environ.get("NEBIUS_PROJECT_ID")
print("Project ID set:", bool(project))

client = ContreeClient(key, project=project)

try:
    me = client.whoami()
    print("\nWHOAMI OK\n", me)
except Exception as e:
    print("\nWHOAMI FAILED:", type(e).__name__, e)

try:
    imgs = client.list_images(tagged=True, limit=20)
    print("\nIMAGES OK\n", imgs)
except Exception as e:
    print("\nIMAGES FAILED:", type(e).__name__, e)