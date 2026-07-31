"""Diagnose WikiText-2 download issue."""
import requests
url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
r = requests.get(url, allow_redirects=False, timeout=30)
print(f'Status: {r.status_code}')
loc = r.headers.get('location', 'none')
print(f'Location: {loc}')

# Also try the redirect target
if loc != 'none':
    r2 = requests.get(loc, allow_redirects=True, timeout=30)
    print(f'Redirect status: {r2.status_code}')
    print(f'Redirect URL: {r2.url}')
    print(f'Size: {len(r2.content)}')
    print(f'Type: {r2.headers.get("Content-Type", "unknown")}')
