
import re

def neteja(text):
  text=text.lower()
  text = re.sub(r'[^\w\s]', '', text)

  ###DUBTE
  titol=["los","ocapis"]
  paraules=text.split()
  if paraules[0:2]==titol and paraules[2:4]==titol:
    return "".join(paraules[2:])

  return text