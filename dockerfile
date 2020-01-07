FROM python:3.7

RUN pip install nltk streamlit gensim
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('wordnet')"

WORKDIR /code

CMD [ "bash" ]