# coding: utf8

import collections
import os
import re

import altair as alt
import gensim.downloader as api
import nltk
import pandas as pd
import streamlit as st
from gensim.models import FastText, KeyedVectors
from nltk.corpus import stopwords

from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn


@st.cache
def load_monster():
    return pd.read_csv("data/monster.csv")


@st.cache
def load_words():
    df = pd.read_csv("data/words.csv")
    male_words = set(df["male"])
    female_words = set(df["female"])

    return male_words, female_words


if os.path.exists("w2v.bin"):
    w2v = KeyedVectors.load("w2v.bin")
else:
    with st.spinner("Downloading Embedding"):
        w2v = api.load("glove-wiki-gigaword-100").wv
        w2v.save("w2v.bin")


def evaluate_text(text):
    tokens = nltk.wordpunct_tokenize(text.lower())
    tokens = [w for w in tokens if w in w2v and w not in sw]

    if not tokens:
        return 0

    progress = st.progress(0)
    score = 0

    for i, tok in enumerate(tokens):
        response = w2v.most_similar(positive=["women", tok], negative=["man"], topn=1)
        response = response[0][0]
        relation = wordnet_relation(tok, response)

        if relation == "antonym":
            score += 1.0
        elif not relation:
            score += 0.5

        progress.progress((i + 1) / len(tokens))

    return score / len(tokens)


neutral_words = [
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

w2v_neutral_words = [w for w in neutral_words if w in w2v]


def is_it_job(tokens):
    for token in tokens:
        if token in IT_WORDS:
            return True


male_words, female_words = load_words()

w2v_male_words = [w for w in male_words if w in w2v]
w2v_female_words = [w for w in female_words if w in w2v]


def get_hyponyms(synset):
    hyponyms = set()

    for hyponym in synset.hyponyms():
        hyponyms |= set(get_hyponyms(hyponym))

    return hyponyms | set(synset.hyponyms())


def get_hypernyms(synset):
    hypernyms = set()

    for hyponym in synset.hypernyms():
        hypernyms |= set(get_hypernyms(hyponym))

    return hypernyms | set(synset.hypernyms())


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

        hyponyms = get_hyponyms(syn)
        for hyp in hyponyms:
            for l in hyp.lemmas():
                if l.name() == w2:
                    return "hyp"

        hypernyms = get_hypernyms(syn)
        for hyp in hypernyms:
            for l in hyp.lemmas():
                if l.name() == w2:
                    return "hyp"

    return ""


def get_swn_score(word):
    synsets = list(swn.senti_synsets(word))

    if not synsets:
        return 0

    sn = synsets[0]

    return sn.pos_score() - sn.neg_score()


def word_analogy(words, positive, negative):
    word_analogy = []

    for w in words:
        response = w2v.most_similar(
            positive=[positive, w], negative=[negative], topn=1
        )[0][0]
        word_analogy.append(
            {
                negative: w,
                positive: response,
                "relation": wordnet_relation(w, response),
                negative + "_score": get_swn_score(w),
                positive + "_score": get_swn_score(response),
            }
        )

    df = pd.DataFrame(word_analogy)
    count = df[df["relation"] != None].count()

    return df


sw = stopwords.words("english")
monster_data = load_monster()

section = st.sidebar.selectbox(
    "Section", ["Classic Words", "Text Analysis", "Overall Corpus"]
)

if section == "Classic Words":
    "### Words"
    st.write(male_words)
    st.write(female_words)

    st.write(word_analogy(["king", "boy", "programmer", "nerd", "doctor"], "woman", "man"))
    st.write(word_analogy(w2v_male_words, "woman", "man"))
    st.write(word_analogy(w2v_neutral_words, "woman", "man"))
    # st.write(word_analogy(w2v_female_words, "man", "woman"))


elif section == "Text Analysis":

    example = st.number_input("Example number", 0, len(monster_data) - 1, 0)
    text_input = st.text_area(
        "Job Posting Example", monster_data["job_description"][example]
    )
    st.write("### Imbalance Score", evaluate_text(text_input))

else:
    "### Monster Dataset"
    st.write(monster_data.head())

    "### Most common words"

    progress = st.progress(0)
    avg = 0
    sample_size = st.slider("Sample size", 0, len(monster_data), 1000)
    available_tags = [
        "CC",   # coordinating conjunction
        "CD",   # cardinal digit
        "DT",   # determiner
        "EX",   # existential there (like: "there is" ... think of it like "there exists")
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
        "NNPS", # proper noun, plural 'Americans'
        "PDT",  # predeterminer 'all the kids'
        "POS",  # possessive ending parent's
        "PRP",  # personal pronoun I, he, she
        "PRP$", # possessive pronoun my, his, hers
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

    for i, row in enumerate(monster_data[:sample_size].itertuples()):
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
    df = word_analogy(most_common, "women", "man")
    st.write(df)
