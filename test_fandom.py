import requests
from bs4 import BeautifulSoup
r = requests.get('https://rezero.fandom.com/wiki/Natsuki_Subaru', headers={'User-Agent': 'Mozilla/5.0'})
soup = BeautifulSoup(r.text, 'html.parser')
print("All paragraphs:", len(soup.find_all('p')))
