import requests
from bs4 import BeautifulSoup

url = "https://witchculttranslation.com/arc-1/"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.text, 'html.parser')

content = soup.find('div', {'class': 'entry-content'})
links = []
if content:
    for a in content.find_all('a', href=True):
        href = a['href']
        if 'arc-1' in href or 'chapter' in href:
            links.append(href)

print(f"Found {len(links)} potential links")
for l in links[:5]:
    print(l)
