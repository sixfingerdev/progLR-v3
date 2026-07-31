"""Download WikiText-103 parquet files from HF Hub."""
from huggingface_hub import hf_hub_download
import os

dl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wt103_raw')
os.makedirs(dl_dir, exist_ok=True)

files = [
    'wikitext-103-v1/train-00000-of-00002.parquet',
    'wikitext-103-v1/train-00001-of-00002.parquet',
    'wikitext-103-v1/validation-00000-of-00001.parquet',
    'wikitext-103-v1/test-00000-of-00001.parquet',
]
for f in files:
    print(f'Downloading {f}...', flush=True)
    path = hf_hub_download(repo_id='wikitext', filename=f, repo_type='dataset', local_dir=dl_dir)
    sz = os.path.getsize(path)
    print(f'  {sz:,} bytes', flush=True)
print('Done', flush=True)
