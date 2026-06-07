import re
import unicodedata


def neteja_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')

    text = text.replace(" ", "_")

    text = re.sub(r'[^a-z0-9_]', '', text)

    return text