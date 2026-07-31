"""Download WikiText-103 from pytorch examples repo."""
import urllib.request, os

base = 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-103'
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wikitext-103')
os.makedirs(dst, exist_ok=True)
files = ['train.txt', 'valid.txt', 'test.txt']
for f in files:
    url = base + '/' + f
    path = os.path.join(dst, f)
    try:
        urllib.request.urlretrieve(url, path)
        sz = os.path.getsize(path)
        print(f'{f}: {sz:,} bytes')
    except Exception as e:
        print(f'{f}: {e}')
print('Done')
