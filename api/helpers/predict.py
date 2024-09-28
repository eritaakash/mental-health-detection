import string, nltk 

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('maxent_ne_chunker_tab')
nltk.download('punkt_tab')
nltk.download('stopwords_tab')
nltk.download('wordnet_tab')
nltk.download('words_tab')
nltk.download('maxent_ne_chunker_en')
nltk.download('punkt_en')
nltk.download('stopwords_en')
nltk.download('wordnet_en')
nltk.download('words_en')


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_text(text):
    # 1. To lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Lemmatize
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]

    return ' '.join(tokens)


mapper = {
    0: 'Normal',
    1: 'Depression',
    2: 'Disorder',
    3: 'Suicidal'
}

def predict(model, tfidf, text):
    text = clean_text(text)
    text = tfidf.transform([text])
    return mapper[model.predict(text)[0]]