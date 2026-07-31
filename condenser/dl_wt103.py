"""Download WikiText-103 raw zip."""
import os, sys, urllib.request

dl_dir = os.path.dirname(os.path.abspath(__file__))
url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
path = os.path.join(dl_dir, 'wikitext-103-v1.zip')

# Follow redirects
class RedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        print(f'  Redirected to: {newurl}', flush=True)
        return urllib.request.Request(newurl, data=req.data, headers=req.headers, origin_req_host=req.origin_req_host, unverifiable=True)

opener = urllib.request.build_opener(RedirectHandler())
urllib.request.install_opener(opener)

print(f'Downloading {url}...', flush=True)
urllib.request.urlretrieve(url, path)
print(f'Done: {os.path.getsize(path)} bytes', flush=True)
