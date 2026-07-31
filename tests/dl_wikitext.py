"""Download proper WikiText-2 with correct train/valid/test split."""
import requests, zipfile, io, os, sys

url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
print(f'Downloading from {url}...')
r = requests.get(url, timeout=120, allow_redirects=True)
print(f'Status: {r.status_code}, Size: {len(r.content)} bytes')

if r.status_code != 200 or len(r.content) < 10000:
    print(f'Failed. Trying alternative URL...')
    url = 'https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-2/1.0.0/wikitext-2-v1.zip'
    r = requests.get(url, timeout=120, allow_redirects=True)
    print(f'Status: {r.status_code}, Size: {len(r.content)} bytes')

z = zipfile.ZipFile(io.BytesIO(r.content))
for name in z.namelist():
    print(f'  {name}')

prefix = 'wikitext-2/'
for fname in ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens']:
    with open(fname, 'wb') as f:
        f.write(z.read(prefix + fname))
    print(f'{fname}: {os.path.getsize(fname)} bytes OK')
print('Done')
