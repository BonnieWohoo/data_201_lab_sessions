
import pandas as pd
import numpy as np
import nltk
import re
from collections import Counter, defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# ==================================================
# 1. CONFIGURATION
# ==================================================

INPUT_FILE = "../youtube.csv"
OUTPUT_FILE = "youtube_comments_analyzed.csv"
THEME_FILE = "youtube_main_themes.csv"

COMMENT_COLUMN = "ytAttributedStringHost"

NRC_FILE = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

# ==================================================
# 2. DOWNLOAD NLTK RESOURCES
# ==================================================

nltk.download("vader_lexicon")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

tokenizer = RegexpTokenizer(r"[a-zA-Z]+")

# ==================================================
# 3. LOAD CSV
# ==================================================

df = pd.read_csv(INPUT_FILE)

df = df.dropna(subset=[COMMENT_COLUMN]).copy()

df[COMMENT_COLUMN] = df[COMMENT_COLUMN].astype(str)

print("Total comments:", len(df))

# ==================================================
# 4. LOAD NRC EMOTION LEXICON
# ==================================================

# NRC file format:
# word    emotion    association (0 or 1)

emotion_categories = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust"
]

nrc_lexicon = defaultdict(set)

nrc_df = pd.read_csv(
    NRC_FILE,
    sep="\t",
    header=None,
    names=["word", "emotion", "association"],
    encoding="utf-8"
)

for _, row in nrc_df.iterrows():

    word = str(row["word"]).lower()
    emotion = str(row["emotion"]).lower()

    if (
        row["association"] == 1
        and emotion in emotion_categories
    ):
        nrc_lexicon[word].add(emotion)

print("NRC emotion lexicon loaded.")

# ==================================================
# 5. POLARITY ANALYSIS USING VADER
# ==================================================

sia = SentimentIntensityAnalyzer()

def polarity_analysis(comment):

    score = sia.polarity_scores(comment)["compound"]

    if score >= 0.05:
        label = "positive"

    elif score <= -0.05:
        label = "negative"

    else:
        label = "neutral"

    return pd.Series([score, label])


df[["polarity_score", "polarity_label"]] = (
    df[COMMENT_COLUMN].apply(polarity_analysis)
)

# ==================================================
# 6. SUBJECTIVITY ANALYSIS
# ==================================================

# Custom opinion and evaluation word list.
# This is a heuristic, not a validated NLTK model.

subjective_words = {
# Opinions and beliefs
    "think", "believe", "feel", "suppose",
    "assume", "guess", "imagine", "consider",
    "argue", "suggest", "recommend", "prefer",

    # Personal opinions
    "personally", "opinion", "perspective",
    "view", "agree", "disagree", "support",
    "oppose", "favor", "favour",

    # Positive evaluations
    "amazing", "wonderful", "excellent", "great",
    "fantastic", "awesome", "beautiful", "brilliant",
    "perfect", "incredible", "outstanding", "helpful",

    # Negative evaluations
    "terrible", "awful", "horrible", "stupid",
    "ridiculous", "useless", "boring", "disappointing",
    "worst", "bad", "poor", "annoying",

    # Emotional expressions
    "love", "hate", "enjoy", "dislike",
    "happy", "sad", "angry", "excited",
    "afraid", "worried", "hope", "fear",

    # Judgments
    "should", "must", "better", "best",
    "wrong", "right", "unfair", "unbelievable",
    "valuable", "important", "interesting",
    "unnecessary", "necessary"
}

def subjectivity_analysis(comment):

    words = tokenizer.tokenize(comment.lower())

    if not words:
        return pd.Series([0.0, "objective"])

    matches = sum(
        word in subjective_words for word in words
    )

    score = matches / len(words)

    label = (
        "subjective" if score >= 0.05
        else "objective"
    )

    return pd.Series([score, label])


df[["subjectivity_score", "subjectivity_label"]] = (
    df[COMMENT_COLUMN].apply(subjectivity_analysis)
)

# ==================================================
# 7. OBJECTIVITY ANALYSIS
# ==================================================

def objectivity_analysis(comment):

    subjectivity = subjectivity_analysis(comment).iloc[0]

    score = 1 - subjectivity

    label = (
        "objective" if score >= 0.95
        else "subjective"
    )

    return pd.Series([score, label])


df[["objectivity_score", "objectivity_label"]] = (
    df[COMMENT_COLUMN].apply(objectivity_analysis)
)

# ==================================================
# 8. EMOTION ANALYSIS USING NRC LEXICON
# ==================================================

def emotion_analysis(comment):

    words = tokenizer.tokenize(comment.lower())

    emotion_counts = Counter()

    for word in words:

        if word in nrc_lexicon:

            for emotion in nrc_lexicon[word]:

                emotion_counts[emotion] += 1

    total_words = max(len(words), 1)

    # Normalize emotion scores by total words
    emotion_scores = {
        emotion: emotion_counts[emotion] / total_words
        for emotion in emotion_categories
    }

    # Identify dominant emotion
    if sum(emotion_counts.values()) == 0:

        dominant_emotion = "no dominant emotion"

    else:

        dominant_emotion = max(
            emotion_counts,
            key=emotion_counts.get
        )

    return pd.Series([
        emotion_scores["anger"],
        emotion_scores["anticipation"],
        emotion_scores["disgust"],
        emotion_scores["fear"],
        emotion_scores["joy"],
        emotion_scores["sadness"],
        emotion_scores["surprise"],
        emotion_scores["trust"],
        dominant_emotion
    ])


emotion_columns = [
    "anger_score",
    "anticipation_score",
    "disgust_score",
    "fear_score",
    "joy_score",
    "sadness_score",
    "surprise_score",
    "trust_score",
    "dominant_emotion"
]

df[emotion_columns] = df[COMMENT_COLUMN].apply(
    emotion_analysis
)

# ==================================================
# 9. THEME IDENTIFICATION USING NLTK BIGRAMS
# ==================================================

def clean_text(comment):

    words = tokenizer.tokenize(comment.lower())

    words = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    return words


df["cleaned_words"] = df[COMMENT_COLUMN].apply(clean_text)

# Extract two-word phrases from all comments
all_bigrams = []

for words in df["cleaned_words"]:

    bigrams = list(nltk.bigrams(words))

    all_bigrams.extend(bigrams)

# Calculate phrase frequencies
bigram_frequency = Counter(all_bigrams)

# Get 10 most frequent two-word themes
top_themes = bigram_frequency.most_common(10)

theme_results = []

for theme, frequency in top_themes:

    theme_results.append({
        "main_theme": " ".join(theme),
        "frequency": frequency
    })

theme_df = pd.DataFrame(theme_results)

theme_df.to_csv(
    THEME_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ==================================================
# 10. ASSIGN MAIN THEME TO EACH COMMENT
# ==================================================

top_theme_set = {
    theme for theme, frequency in top_themes
}

def assign_theme(words):

    comment_bigrams = list(nltk.bigrams(words))

    matching_themes = [
        bigram for bigram in comment_bigrams
        if bigram in top_theme_set
    ]

    if not matching_themes:
        return "Uncategorized"

    counts = Counter(matching_themes)

    dominant_theme = counts.most_common(1)[0][0]

    return " ".join(dominant_theme)


df["main_theme"] = df["cleaned_words"].apply(
    assign_theme
)



# ==========================================
# EMOTION ANALYSIS USING NRC LEXICON
# ==========================================

def emotion_analysis(comment):

    words = tokenizer.tokenize(comment.lower())

    emotion_counts = Counter()

    for word in words:

        if word in nrc_lexicon:

            for emotion in nrc_lexicon[word]:

                emotion_counts[emotion] += 1

    total_words = max(len(words), 1)

    # Calculate normalized emotion scores
    emotion_scores = {
        emotion: emotion_counts[emotion] / total_words
        for emotion in emotion_categories
    }

    # Identify dominant emotion
    if sum(emotion_counts.values()) == 0:

        main_emotion = "no dominant emotion"

    else:

        main_emotion = max(
            emotion_scores,
            key=emotion_scores.get
        )

    return pd.Series([
        emotion_scores["anger"],
        emotion_scores["anticipation"],
        emotion_scores["disgust"],
        emotion_scores["fear"],
        emotion_scores["joy"],
        emotion_scores["sadness"],
        emotion_scores["surprise"],
        emotion_scores["trust"],
        main_emotion
    ])


emotion_columns = [
    "anger_score",
    "anticipation_score",
    "disgust_score",
    "fear_score",
    "joy_score",
    "sadness_score",
    "surprise_score",
    "trust_score",
    "main_emotion"
]

df[emotion_columns] = df[COMMENT_COLUMN].apply(
    emotion_analysis
)


# ==================================================
# 11. SAVE FINAL RESULTS
# ==================================================

df.drop(columns=["cleaned_words"]).to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ==================================================
# 12. DISPLAY RESULTS
# ==================================================

print("\nANALYSIS COMPLETED!")

print("\nTOP 10 MAIN THEMES:")

print(theme_df.to_string(index=False))

print("\nPOLARITY DISTRIBUTION:")

print(df["polarity_label"].value_counts())

print("\nDOMINANT EMOTION DISTRIBUTION:")

print(df["dominant_emotion"].value_counts())

print("\nSUBJECTIVITY DISTRIBUTION:")

print(df["subjectivity_label"].value_counts())

print("\nOUTPUT FILE:", OUTPUT_FILE)

print("\nFIRST FIVE ANALYZED COMMENTS:")

print(df.head())
