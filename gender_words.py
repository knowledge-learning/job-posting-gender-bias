# coding: utf8

import collections
import os
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import gensim.downloader as api
import nltk
from gensim.models import FastText, KeyedVectors
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords, wordnet
from scipy.spatial.distance import cosine
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

NEUTRAL_WORDS = [
    "proven",
    "sound",
    "solid",
    "run",
    "steer",
    "deliver",
    "energy",
    "motor",
    "run",
    "manage",
    "grow",
    "research",
    "testing",
    "scrutiny",
    "people",
    "team",
    "actions",
    "moves",
    "attractive",
    "fair",
    "results-oriented",
]


def tab(section):
    import collections

    if not hasattr(tab, '__tabs__'):
        tab.__tabs__ = collections.defaultdict(dict)

        def run(sec, *args, **kwargs):
            func = st.sidebar.selectbox(sec, list(tab.__tabs__[sec]))
            func = tab.__tabs__[sec][func]
            func(*args, **kwargs)

        tab.run = run

    def wrapper(func):
        name = " ".join(s.title() for s in func.__name__.split("_"))
        tab.__tabs__[section][name] = func
        return func

    return wrapper


class App:
    def __init__(self):
        self.monster_data = self._load_monster()

        w2v = self._load_embedding()
        male_words, female_words = self._load_words()

        self.w2v = w2v
        self.w2v_male_words = [w for w in male_words if w in w2v]
        self.w2v_female_words = [w for w in female_words if w in w2v]
        self.w2v_neutral_words = [w for w in NEUTRAL_WORDS if w in self.w2v]

        self.sw = stopwords.words("english")

    @st.cache
    def _load_monster(self):
        return pd.read_csv("data/monster.csv")

    @st.cache
    def _load_words(self):
        df = pd.read_csv("data/words.csv")
        male_words = set(df["male"])
        female_words = set(df["female"])

        return male_words, female_words

    @st.cache(allow_output_mutation=True)
    def _load_embedding(self, ):
        if os.path.exists("w2v.bin"):
            w2v = KeyedVectors.load("w2v.bin")
        else:
            with st.spinner("Downloading Embedding"):
                w2v = api.load("glove-wiki-gigaword-100").wv
                w2v.save("w2v.bin")
        return w2v

    @staticmethod
    def get_hyponyms(synset):
        hyponyms = set()

        for hyponym in synset.hyponyms():
            hyponyms |= set(App.get_hyponyms(hyponym))

        return hyponyms | set(synset.hyponyms())

    @staticmethod
    def get_hypernyms(synset):
        hypernyms = set()

        for hyponym in synset.hypernyms():
            hypernyms |= set(App.get_hypernyms(hyponym))

        return hypernyms | set(synset.hypernyms())

    @staticmethod
    def wordnet_relation(w1, w2):
        print(w1, w2)

        if w1.startswith(w2) or w2.startswith(w1):
            return "synonim"

        w1_synsets = set(wordnet.synsets(w1))
        w2_synsets = set(wordnet.synsets(w2))

        if w1_synsets & w2_synsets:
            return "synonim"

        for syn in w1_synsets:
            for l in syn.lemmas():
                for a in l.antonyms():
                    if a.name() == w2:
                        return "antonym"

            hyponyms = App.get_hyponyms(syn)
            for hyp in hyponyms:
                for l in hyp.lemmas():
                    if l.name() == w2:
                        return "hyp"

            hypernyms = App.get_hypernyms(syn)
            for hyp in hypernyms:
                for l in hyp.lemmas():
                    if l.name() == w2:
                        return "hyp"

        return ""

    def get_swn_score(self, word):
        synsets = list(swn.senti_synsets(word))

        if not synsets:
            return 0

        sn = synsets[0]

        return sn.pos_score() - sn.neg_score()

    def get_swn_mean_score(self, word):
        synsets = list(swn.senti_synsets(word))

        if not synsets:
            return 0

        return sum((sn.pos_score() - sn.neg_score() for sn in synsets)) / len(synsets)

    def get_alignment(self, gold_source, gold_target, query_source, query_target):
        gold_source = self.w2v[gold_source]
        query_source = self.w2v[query_source]
        gold_target = self.w2v[gold_target]
        query_target = self.w2v[query_target]
        return 1 - cosine(gold_target - gold_source, query_target - query_source)

    def word_analogy(self, words, positive, negative, categorical=False):
        word_analogy = []

        for w in words:
            query = self.w2v.most_similar(
                positive=[positive, w], negative=[negative], topn=1
            )
            for response in query:
                response = response[0]
                info = {
                    negative: w,
                    positive: response,
                    negative + "_mean_score": self.get_swn_mean_score(w),
                    positive + "_mean_score": self.get_swn_mean_score(response),
                    'alignment': self.get_alignment(positive, w, negative, response),
                    'alignment-dual': self.get_alignment(positive, negative, w, response),
                }
                relation = self.wordnet_relation(w, response)
                if categorical:
                    if relation:
                        info[relation] = 1
                else:
                    info['relation'] = relation
                word_analogy.append(info)

        df = pd.DataFrame(word_analogy)
        return df

    def score_similarity(self, words, *labels):
        items = []
        for word in words:
            info = {'word': word}
            for l in labels:
                info[l] = self.w2v.similarity(word, l)
            items.append(info)
        return pd.DataFrame(items)

    def score_triangle(self, words, categorical=False):
        items = []
        for word in words:
            woman = self.w2v.most_similar(
                positive=['woman', word], negative=['man'], topn=1
            )[0][0]
            man = self.w2v.most_similar(
                positive=['man', word], negative=['woman'], topn=1
            )[0][0]

            info = {}
            info['word'] = word
            info['woman'] = woman
            info['man'] = man

            info['crossterm_similarity'] = self.w2v.similarity(woman, man)
            info['woman_similarity'] = self.w2v.similarity(word, woman)
            info['man_similarity'] = self.w2v.similarity(word, man)

            crossterm_relation = self.wordnet_relation(woman, man)
            woman_relation = self.wordnet_relation(word, woman)
            man_relation = self.wordnet_relation(word, man)

            if categorical:
                if crossterm_relation: info[f'{crossterm_relation}_crossterm'] = 1
                if woman_relation: info[f'{woman_relation}_woman'] = 1
                if man_relation: info[f'{man_relation}_man'] = 1
            else:
                info['crossterm_relation'] = self.wordnet_relation(woman, man)
                info['woman_relation'] = self.wordnet_relation(word, woman)
                info['man_relation'] = self.wordnet_relation(word, man)

            info['word_sentiment'] = self.get_swn_mean_score(word)
            info['woman_sentiment'] = self.get_swn_mean_score(woman)
            info['man_sentiment'] = self.get_swn_mean_score(man)
            
            items.append(info)
        return pd.DataFrame(items)

    def run(self):
        tab.run('section', self)

    @tab('section')
    def similarity(self):
        st.write('### Male')
        st.write(self.score_similarity(self.w2v_male_words, 'man', 'woman'))
        st.write('### Female')
        st.write(self.score_similarity(self.w2v_female_words, 'man', 'woman'))
        st.write('### Neutral')
        st.write(self.score_similarity(self.w2v_neutral_words, 'man', 'woman'))

    @tab('section')
    def triangle(self):
        st.write('### Selected')
        st.write(self.score_triangle(["king", "boy", "programmer", "nerd", "doctor"]))
        st.write('### Male')
        st.write(self.score_triangle(self.w2v_male_words))
        st.write('### Female')
        st.write(self.score_triangle(self.w2v_female_words))
        st.write('### Neutral')
        st.write(self.score_triangle(self.w2v_neutral_words))

    @tab('section')
    def classic_words(self):
        st.show(set(self.w2v_male_words))
        st.show(set(self.w2v_female_words))

        st.write('### Selected')
        st.write(self.word_analogy(
            ["king", "boy", "programmer", "nerd", "doctor"], "woman", "man"))
        st.write(self.word_analogy(
            ["king", "boy", "programmer", "nerd", "doctor"], "man", "woman"))
        st.write('### Male')
        st.write(self.word_analogy(self.w2v_male_words, "woman", "man"))
        st.write(self.word_analogy(self.w2v_male_words, "man", "woman"))
        st.write('### Female')
        st.write(self.word_analogy(self.w2v_female_words, "man", "woman"))
        st.write(self.word_analogy(self.w2v_female_words, "woman", "man"))
        st.write('### Neutral')
        st.write(self.word_analogy(self.w2v_neutral_words, "woman", "man"))
        st.write(self.word_analogy(self.w2v_neutral_words, "man", "woman"))
        # st.write(word_analogy(w2v_female_words, "man", "woman"))

    @tab('section')
    def decision_tree(self):

        def get_data(words):
            male2female = self.word_analogy(words, "woman", "man", True)
            male2female = male2female[male2female.columns[2:]]
            male2female = male2female.add_suffix('_A')

            female2male = self.word_analogy(words, "woman", "man", True)
            female2male = female2male[female2male.columns[2:]]
            female2male = female2male.add_suffix('_B')

            return pd.concat((male2female, female2male), axis=1)

        def get_triangular_data(words):
            data = self.score_triangle(words, True)
            return data[data.columns[3:]]

        def build_data(shuffle=True):
            # X_male = get_data(self.w2v_male_words)
            # X_female = get_data(self.w2v_female_words)
            # X_neutral = get_data(self.w2v_neutral_words)

            X_male = get_triangular_data(self.w2v_male_words)
            X_female = get_triangular_data(self.w2v_female_words)
            X_neutral = get_triangular_data(self.w2v_neutral_words)

            X = pd.concat((X_male, X_female, X_neutral))
            X.fillna(0, inplace=True)
            st.show(X)

            X = X.to_numpy()

            y_male = np.ones(len(X_male)) * -1.0
            y_female = np.ones(len(X_female))
            y_neutral = np.zeros(len(X_neutral))
            y = np.concatenate((y_male, y_female, y_neutral))

            if shuffle:
                np.random.seed(0)
                indexes = np.arange(len(X))
                np.random.shuffle(indexes)

                X = X[indexes]
                y = y[indexes]

            return X, y

        X, y = build_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

        st.show(X, y)

        model = RandomForestClassifier(max_depth=5)
        model.fit(X_train, y_train)
        # st.show(plot_tree(model))
        st.show(model.score(X_train, y_train))
        st.show(model.score(X_test, y_test))

        st.show(np.c_[y_test, model.predict(X_test)])

    @tab('section')
    def text_analysis(self):
        example = st.number_input(
            "Example number", 0, len(self.monster_data) - 1, 0)
        text_input = st.text_area(
            "Job Posting Example", self.monster_data["job_description"][example]
        )
        st.write("### Imbalance Score", self.evaluate_text(text_input))

    def evaluate_text(self, text):
        tokens = nltk.wordpunct_tokenize(text.lower())
        tokens = [w for w in tokens if w in self.w2v and w not in self.sw]

        if not tokens:
            return 0

        progress = st.progress(0)
        score = 0

        for i, tok in enumerate(tokens):
            response = self.w2v.most_similar(
                positive=["women", tok], negative=["man"], topn=1)
            response = response[0][0]
            relation = self.wordnet_relation(tok, response)

            if relation == "antonym":
                score += 1.0
            elif not relation:
                score += 0.5

            progress.progress((i + 1) / len(tokens))

        return score / len(tokens)

    @tab('section')
    def overall_corpus(self):
        "### Monster Dataset"
        st.write(self.monster_data.head())

        "### Most common words"

        progress = st.progress(0)
        avg = 0
        sample_size = st.slider(
            "Sample size", 0, len(self.monster_data), 1000)
        available_tags = [
            "CC",   # coordinating conjunction
            "CD",   # cardinal digit
            "DT",   # determiner
            # existential there (like: "there is" ... think of it like "there exists")
            "EX",
            "FW",   # foreign word
            "IN",   # preposition/subordinating conjunction
            "JJ",   # adjective 'big'
            "JJR",  # adjective, comparative 'bigger'
            "JJS",  # adjective, superlative 'biggest'
            "LS",   # list marker 1)
            "MD",   # modal could, will
            "NN",   # noun, singular 'desk'
            "NNS",  # noun plural 'desks'
            "NNP",  # proper noun, singular 'Harrison'
            "NNPS",  # proper noun, plural 'Americans'
            "PDT",  # predeterminer 'all the kids'
            "POS",  # possessive ending parent's
            "PRP",  # personal pronoun I, he, she
            "PRP$",  # possessive pronoun my, his, hers
            "RB",   # adverb very, silently,
            "RBR",  # adverb, comparative better
            "RBS",  # adverb, superlative best
            "RP",   # particle give up
            "TO",   # to go 'to' the store.
            "UH",   # interjection errrrrrrrm
            "VB",   # verb, base form take
            "VBD",  # verb, past tense took
            "VBG",  # verb, gerund/present participle taking
            "VBN",  # verb, past participle taken
            "VBP",  # verb, sing. present, non-3d take
            "VBZ",  # verb, 3rd person sing. present takes
            "WDT",  # wh-determiner which
            "WP",   # wh-pronoun who, what
            "WP$",  # possessive wh-pronoun whose
            "WRB",  # wh-abverb where, when
        ]
        pos_tags = st.multiselect(
            "Part-Of-Speech to include", available_tags, ["JJ", "VB"]
        )

        sw = stopwords.words()
        word_counter = collections.Counter()

        for i, row in enumerate(self.monster_data[:sample_size].itertuples()):
            text = row.job_description
            tokens = nltk.wordpunct_tokenize(text.lower())
            tokens = nltk.pos_tag(tokens)
            tokens = [
                w for w, tag in tokens if w.isalpha() if tag in pos_tags
            ]  # and w not in sw]

            word_counter.update(tokens)
            progress.progress((i + 1) / sample_size)

        for w in sw:
            if w in word_counter:
                word_counter.pop(w)

        most_common = [w[0] for w in word_counter.most_common(100)]
        df = self.word_analogy(most_common, "women", "man")
        st.write(df)


app = App()
app.run()
