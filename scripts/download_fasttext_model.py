"""
Downloads the fasttext model for language identification.
"""
import requests
import os
from tqdm import tqdm

def download_model():
    """
    Downloads the fasttext language identification model.
    """
    model_path = "models/lid.176.bin"
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return

    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    print(f"Downloading model from {url}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(model_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)
        
        if total_size != 0 and bar.n != total_size:
            print("ERROR, something went wrong during download")
        else:
            print(f"Model downloaded to {model_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    download_model()
