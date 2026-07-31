"""Download WikiText-103 for benchmark."""
from datasets import load_dataset
import os, re

print("Loading WikiText-103 train (streaming, first 50000 lines)...")
ds = load_dataset('mattdangerw/wikitext-103-raw', split='train', streaming=True)

train_lines = []
total_chars = 0
for i, item in enumerate(ds):
    train_lines.append(item['text'])
    total_chars += len(item['text'])
    if (i+1) % 10000 == 0:
        print(f'  {i+1} lines, {total_chars/1e6:.0f}M chars...')
    if total_chars > 10_000_000 and len(train_lines) > 50000:
        break

with open('wt103.train.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(train_lines))
print(f'Train: {os.path.getsize("wt103.train.txt")/1e6:.1f}MB')

# Test set
ds_test = load_dataset('mattdangerw/wikitext-103-raw', split='test', streaming=True)
test_lines = [item['text'] for item in ds_test]
with open('wt103.test.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(test_lines))
print(f'Test: {os.path.getsize("wt103.test.txt")/1e6:.1f}MB')

# Validation
ds_val = load_dataset('mattdangerw/wikitext-103-raw', split='validation', streaming=True)
val_lines = [item['text'] for item in ds_val]
with open('wt103.valid.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(val_lines))
print(f'Valid: {os.path.getsize("wt103.valid.txt")/1e6:.1f}MB')
print('Done')
