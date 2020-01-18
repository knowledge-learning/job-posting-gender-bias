FROM python:3.7

RUN pip install nltk streamlit gensim
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('wordnet')"
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
RUN python -c "import nltk; nltk.download('sentiwordnet')"
RUN pip install sklearn, matplotlib

WORKDIR /code

CMD [ "bash" ]