"""Download correct WikiText-2 test set."""
from datasets import load_dataset
import os

print("Loading test split...")
ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test', streaming=True)
texts = []
for i, item in enumerate(ds):
    texts.append(item['text'])
    if (i+1) % 1000 == 0:
        print(f'  {i+1} lines...')

with open('wiki.test.tokens', 'w', encoding='utf-8') as f:
    f.write(''.join(texts))
print(f'Done: {os.path.getsize("wiki.test.tokens")} bytes, {len(texts)} lines')
