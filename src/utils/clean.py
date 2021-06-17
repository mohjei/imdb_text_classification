
import nltk
import re
from nltk.corpus import stopwords
from src.utils.fix_characters import appo
nltk.download('stopwords')


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', text)  # clean url
    text = re.sub(r'#(\w+)', '', text)  # clean hashes
    text = re.sub(r'@(\w+)', '', text)  # clean @
    text = re.sub(r'<[^>]+>', '', text)  # clean tags
    text = re.sub(r'\d+', '', text)  # clean digits
    text = re.sub(r'==(\w+)', '', text)  # clean hashes
    text = re.sub(r'[,!@\'\"?\.$%_&#*+-:;]', '', text)  # clean punctuation
    text = [appo[word] if word in appo else word for word in text.split()]  #
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    text = [w for w in text if not w in stop_words]
    # filter out short tokens
    text = [word for word in text if len(word) > 1]

    return text
