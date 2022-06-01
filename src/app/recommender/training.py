import os
import pickle
import random
import re
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords

random.seed(100)

dataFile = sys.argv[1]
if not os.path.isfile(dataFile):
    print("File '" + dataFile + "' does not exits.")
    sys.exit(1)

print("Num GPUs available: ", len(tf.config.experimental.list_physical_devices("GPU")))

# LOAD AND PREPROCESS THE DATASET
df = pd.read_csv(dataFile, sep=",")
# df = pd.read_csv("doc2vec.csv",sep=',')
df.columns = ["text", "id"]
# df = df[0:1000]

df = df.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile(r"[/(){}_\[\]\|@,;]")
BAD_SYMBOLS_RE = re.compile(r"[^0-9a-zäöüß ]")
STOPWORDS = (
    set(stopwords.words("german"))
    .union(set(stopwords.words("english")))
    .union(
        {
            "https",
            "http",
            "lernen",
            "wwwyoutubecom",
            "video",
            "videos",
            "erklärt",
            "einfach",
            "nachhilfe",
            "bitly",
            "online",
            "ordne",
            "mehr",
            "a",
            "hilfe",
            "amznto",
            "wwwfacebookcom",
            "zahlen",
            "b",
            "schule",
            "kostenlos",
            "c",
            "facebook",
            "klasse",
            "unterricht",
            "finden",
            "de",
            "richtigen",
            "themen",
            "fragen",
            "gibt",
            "studium",
            "richtig",
            "richtige",
            "wissen",
            "onlinenachhilfe",
            "finde",
            "schüler",
            "learn",
            "uni",
            "teil",
            "e",
            "weitere",
            "co",
            "aufgaben",
            "twittercom",
            "bild",
            "verben",
            "einzelnen",
            "wwwinstagramcom",
            "berechnen",
            "youtube",
            "twitter",
            "media",
            "lernvideo",
            "quiz",
            "abitur",
            "schnell",
            "thema",
            "free",
            "zeit",
            "website",
            "angaben",
            "erklärvideo",
            "social",
            "bestandteile",
            "mal",
            "top",
            "findest",
            "tet",
            "beispiel",
            "spaß",
            "immer",
            "urhebern",
            "zwei",
            "beim",
            "viele",
            "lizenzbedingungen",
            "seite",
            "kurze",
            "besser",
            "begriffe",
            "infos",
            "la",
            "bzw",
            "plattform",
            "nachhilfeunterricht",
            "lernhilfe",
            "nachhilfelehrer",
            "wurde",
            "onlinehilfe",
            "wer",
            "onlinelehrer",
            "findet",
            "wwwtutoryde",
            "kürze",
            "ordnen",
            "dokument",
            "onlineunterricht",
            "umsonst",
            "world",
            "us",
            "merkhilfe",
            "bereitstellung",
            "schoolseasy",
            "kanal",
            "kostenlose",
            "instagram",
            "schülernachhilfe",
        }
    )
)


def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(" ", text)
    text = BAD_SYMBOLS_RE.sub("", text)
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text


# print (df['text'][:5])
df["text"] = df["text"].apply(clean_text)
df["text"] = df["text"].str.replace(r"\d+", "")
# print (df['text'][:5])


# TOKENIZE AND CLEAN TEXT
# The maximum number of words to be used. (most frequent)
MAX_DICT_SIZE = 20000

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=MAX_DICT_SIZE, filters=r'!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True
)
tokenizer.fit_on_texts(df["text"].values)
word_trans = tokenizer.word_index
print("Found %s unique tokens." % len(word_trans))

X = tokenizer.texts_to_sequences(df["text"].values)

pairs = []
words = []

for d in range(len(X)):
    for w in X[d]:
        words.append(w)

words = set(words)
print("Num of words: ", len(words))

word_index = {word: idx for idx, word in enumerate(words)}
index_word = {idx: word for word, idx in word_index.items()}

for d in range(len(X)):
    for w in X[d]:
        pairs.append((d, word_index[w]))

print("Num of pairs: ", len(pairs))
pairs_set = set(pairs)


def generate_batch(pairs, n_positive=50, negative_ratio=1.0):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))

    neg_label = -1

    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (doc_id, word_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (doc_id, word_id, 1)

        # Increment idx by 1
        idx += 1

        # Add negative examples until reach batch size
        while idx < batch_size:

            # random selection
            random_doc = random.randrange(len(X))
            random_word = random.randrange(len(word_index))

            # Check to make sure this is not a positive example
            if (random_doc, random_word) not in pairs_set:
                # Add to batch and increment index
                batch[idx, :] = (random_doc, random_word, neg_label)
                idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {"doc": batch[:, 0], "word": batch[:, 1]}, batch[:, 2]


def embedding_model(embedding_size=20):
    # Both inputs are 1-dimensional
    doc = tf.keras.layers.Input(name="doc", shape=[1])
    word = tf.keras.layers.Input(name="word", shape=[1])

    # Embedding the doc (shape will be (None, 1, 50))
    doc_embedding = tf.keras.layers.Embedding(
        name="doc_embedding", input_dim=len(X), output_dim=embedding_size
    )(doc)

    # Embedding the word (shape will be (None, 1, 50))
    word_embedding = tf.keras.layers.Embedding(
        name="word_embedding", input_dim=len(word_index), output_dim=embedding_size
    )(word)

    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = tf.keras.layers.Dot(name="dot_product", normalize=True, axes=2)(
        [doc_embedding, word_embedding]
    )

    # Reshape to be a single number (shape will be (None, 1))
    merged = tf.keras.layers.Reshape(target_shape=[1])(merged)

    model = tf.keras.models.Model(inputs=[doc, word], outputs=merged)
    model.compile(optimizer="Adam", loss="mse")

    return model


# DEFINE THE MODEL
model = embedding_model()
model.summary()

# PREPARE DATA BATCHES
# gen = generate_batch(pairs, n_positive = len(pairs), negative_ratio = 1)
gen = generate_batch(pairs, n_positive=100000, negative_ratio=1)

# DO THE TRAINING
# n_positive = 512
n_positive = 100000
h = model.fit(gen, epochs=20, steps_per_epoch=len(pairs) // n_positive, verbose=2)

model.save(dataFile.replace(".csv", "-embed.h5"))

# store document ids
with open(dataFile.replace(".csv", "-id.pickle"), "wb") as handle:
    pickle.dump(df["id"], handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done :-)")
