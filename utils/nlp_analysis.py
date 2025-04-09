import sys #provides access to system-related information 

#NLP 
import spacy #Industrial-strength NLP library for tokenisation, named entity recognition, and dependency parsing
spacy.cli.download("en_core_web_sm")
import nltk #natural language toolkit -> NLP library for tokenisation, stemming, and corpus handling 
from nltk.tokenize import sent_tokenize #function to split text into sentences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
from transformers import pipeline #Pre-trained NLP models for text classification, summarisation, translation etc
import bertopic
from bertopic import BERTopic 
from sentence_transformers import SentenceTransformer
from umap import UMAP #required for UMAP-based dimensionality reduction 
import hdbscan #required for hierarchical density-based clustering
import torch #deep learning framework for building and training neural networks 
 
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 

#Data Structures 
import pandas as pd #provides data structures (DataFrame, Series) for efficient data 
import numpy as np #support for large multi-dim arrays

#Others 
import time 
import re #regular expressions: used for pattern matching in strings 
from collections import defaultdict
from collections import Counter
import os #provides functions for interacting with operating system (file paths, environment variables etc)

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

from wordcloud import WordCloud 
import matplotlib.pyplot as plt

#Retrieve the right section 
def retrieve_corpus(dictionary, section_names): 
    """
    Retrieves and concatenates multiple sections from the given dictionary
    Args: 
    - Dictionary (dict): contains section names as keys and text as values 
    - section_names (list): list of section names to retrieve (e.g. ["Item 1", "Item 7"])
    Returns: 
    - str: concatenated content of requested sections or message if none are available 
    """

    try: 
        if not isinstance(dictionary, dict): 
            raise TypeError("Error: dictionary must be a dict")
        if not isinstance(section_names, list): 
            raise TypeError("Error: Section_Names must be a list")
        if not all(isinstance(section, str) for section in section_names): 
            raise TypeError("Error: Each section name must be a str")
    
        #List comprehension to pull out for each section in section_names the relevant content. section is "Item 1", "Item 2" etc. section_names is list of these sections
        retrieved_texts = [
            dictionary[section]["content"] if section in dictionary else f"[{section} not found]"
            for section in section_names
        ]

        combined_text = "\n\n---\n\n".join(retrieved_texts)

        return combined_text if any(
            text != f"[{section} not found]" for section, text in zip(section_names, retrieved_texts)
        ) else "No valid sections found."

    except Exception as e: 
        return str(e)

# Basic Text Processing 
# - Tokenisation (Splitting text into words)
# - Lemmatisation (reducing words to their base form) 
# - Word frequency analysis (finding most common words)
def process_text(corpus, top_n=20, remove_stopwords=True): 
    """
    Tokenises, lemmatizes, and performs word frequency analysis on the given text 
    
    Params: 
    - corpus: (str) input text to analyze 
    - top_n: (int) number of most common words to return 

    Returns: 
    - list of tuples with (word, frequency)
    - cleaned, tokenised text (for further NLP task like topic modelling or sentiment analysis)
    """

    #Tokenisation: Extract words only
    tokens = re.findall(r'\b[a-zA-Z]+\b', corpus.lower())

    #Initialise lemmatizer 
    lemmatizer = WordNetLemmatizer()

    #Load stopwords (if removal is enabled) into set 
    stop_words = set(stopwords.words('english')) if remove_stopwords else set() 

    #Lemmatisation and stopword removal 
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    #Word Frequency Analysis 
    word_freq = Counter(processed_tokens)

    #Get top N most common words 
    top_words = word_freq.most_common(top_n)

    return top_words, processed_tokens 

import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure NLTK components are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Optional: install and download POS tagger
nltk.download('averaged_perceptron_tagger')

# Define custom financial stopwords
FINANCIAL_STOPWORDS = set([
    "company", "income", "revenue", "net", "fiscal", "report", "reporting", 
    "financial", "statement", "share", "common", "preferred", "inc", 
    "corporation", "year", "ended", "consolidated", "amount", "table",
    "million", "thousand", "unaudited", "balance", "assets", "liabilities",
    "cash", "flow", "operations", "expense", "cost", "data", "result", "product", "result", "third", "may", "service"
])

def process_text_2(corpus, top_n=20, remove_stopwords=True, filter_pos=False): 
    """
    Tokenises, lemmatizes, and performs word frequency analysis on the given text 
    
    Params: 
    - corpus: (str) input text to analyze 
    - top_n: (int) number of most common words to return 
    - remove_stopwords: (bool) remove standard + financial stopwords
    - filter_pos: (bool) keep only nouns and adjectives

    Returns: 
    - list of tuples with (word, frequency)
    - cleaned, tokenised text (for further NLP task like topic modelling or sentiment analysis)
    """
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', corpus.lower())  # Only words ≥ 3 letters

    lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    stop_words |= FINANCIAL_STOPWORDS  # Add financial stopwords

    # Optional POS tagging
    if filter_pos:
        tagged = nltk.pos_tag(tokens)
        tokens = [word for word, pos in tagged if pos.startswith("N") or pos.startswith("J")]

    # Lemmatize and remove stopwords
    processed_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words
    ]

    word_freq = Counter(processed_tokens)
    top_words = word_freq.most_common(top_n)

    return top_words, processed_tokens

#Named Entity Recognition (NER) with SpaCy

#load SpaCy's financial NER model (or use "en_core_web_sm" for a general model)
nlp = spacy.load("en_core_web_sm") #for finance-specific, use "en_core_web_trf" if available

def extract_named_entities(text, relevant_categories=None): 
    """
    Extract named entities from text by: 
    - Processing using SpaCy model (nlp)
    - Extracting named entities (e.g. companies, locations, dates, financial terms)
    - Group entities by their type (e.g. ORG for organisations, MONEY for financial amounts etc)
    - Removes duplicates before returning the dictionary

    Parameters: 
    - text (str): input corpus 
    - relevant_categories (set): Optional. Set of entity labels to filter 

    Returns: 
    - dict: Extracted named entities categorised by type

    """
    #Default to financial categories if none are provided
    if relevant_categories is None:
        relevant_categories = {"MONEY", "ORG", "GPE", "PERCENT", "DATE"} 

    #Convert categories to uppercase to avoid case issues 
    relevant_categories = {cat.upper() for cat in relevant_categories}

    doc = nlp(text) #process text
    entities = defaultdict(list) #stores entities in a dictionary 

    for ent in doc.ents: 
        if relevant_categories is None or ent.label_ in relevant_categories: 
            entities[ent.label_].append(ent.text)

    result = {label: list(set(mentions)) for label, mentions in entities.items()}  #remove duplicates 

    if not result: 
        print("No relevant named entities found in the text")
    
    return result 

#Smart chunking for long sections 
def chunk_text(text, chunk_size=512, overlap=50): 
    """
    Splits long corpus into smaller overlapping chunks for models with input limits. 
    Overlap ensures no context loss

    Params: 
    - Chunk_size; maximum length of each chunk 
    - overlap: number of words repeated from the previous chunk for continuity 

    Returns: 
    - List of text chunks
    """
    words = text.split()
    chunks = [] #initialise empty list 
    start = 0

    while start < len(words): 
        end = min(start + chunk_size, len(words)) #ensure don't exceed the text length 
        chunks.append(" ".join(words[start:end])) #convert word list back to string 
        start += chunk_size -overlap #move roward whilst keeping overlap
    
    return chunks 

#Smart chunking for long sections 
def chunk_text2(text, chunk_size=256, overlap=56): 
    """
    Splits long corpus into smaller overlapping chunks for models with input limits. 
    Overlap ensures no context loss

    Params: 
    - Chunk_size; maximum length of each chunk 
    - overlap: number of words repeated from the previous chunk for continuity 

    Returns: 
    - List of text chunks
    """
    nltk.download("punkt", quiet=True)
    sentences = sent_tokenize(text) #splits full text into individual sentences

    chunks = [] #hold the final output which is the list of string chunks 
    current_chunk = [] #temporarily holds the list of sentences as we construct each chunk 
    current_word_count = 0 #keeps track of how many words we accumulated in the current chunk
    i = 0 #sentence index

    while i < len(sentences): #main loop: loop thru all sentences by index
        sentence = sentences[i] 
        word_count = len(sentence.split()) #get the next sentence and count it words 

        #if adding this sentence keeps chunk within word limit, just add it in and continue
        if current_word_count + word_count <= chunk_size: 
            current_chunk.append(sentence)
            current_word_count += word_count
            i += 1 
        
        #if it goes over the word target, the current chunk is done, so join its sentences and add to final chunks 
        else: 
            chunks.append(" ".join(current_chunk))

            #Build the overlap (backtrack)
            overlap_chunk = []
            overlap_count = 0 
            for sent in reversed(current_chunk): 
                wc = len(sent.split())
                if overlap_count + wc <= overlap:
                    overlap_chunk.insert(0, sent)
                    overlap_count += wc
                else: 
                    break 
            current_chunk = overlap_chunk
            current_word_count = overlap_count
    
    if current_chunk: 
        chunks.append(" ".join(current_chunk))
    
    return chunks
    #words = text.split()
    #chunks = [] #initialise empty list 
    #start = 0

    #while start < len(words): 
    #    end = min(start + chunk_size, len(words)) #ensure don't exceed the text length 
    #    chunks.append(" ".join(words[start:end])) #convert word list back to string 
    #    start += chunk_size -overlap #move roward whilst keeping overlap
    
    #return chunks 

def chunk_by_sentences(text, sentences_per_chunk=5): 
    """
    Splits a long text into chunks of N sentences each. 

    Parameters: 
    - text (str): input corpus 
    - sentences_per_chunk (int): number of sentences per chunk 

    Returns: 
    - List (str): list of sentence-based chunks 
    """
    nltk.download("punkt", quiet=True)
    sentences = sent_tokenize(text)
    chunks=[]

    for i in range(0, len(sentences), sentences_per_chunk): 
        chunk = " ".join(sentences[i: i+sentences_per_chunk])
        chunks.append(chunk)
    return chunks 

#Topic modelling with BERTopic: 
from sklearn.feature_extraction import text 
from bertopic.representation import MaximalMarginalRelevance

def preprocess_text(text):
    """Remove numbers, financial terms, and stopwords before tokenization."""
    text = re.sub(r'\b\d+\b', '', text)  # Remove numbers
    text = re.sub(r'\b(form|net sales|gross margin|fiscal year|billion|quarterly report|effective tax|ipad|wearables|applecare|iphone|service|assurance|tax|demand)\b', '', text, flags=re.IGNORECASE)
    
    # Manual stopword removal before vectorization
    total_stopwords = {
        "company", "companys", "business", "products", "services", "year", "2024", "2023", 
        "apple", "market", "product", "financial", "including", "form", "net sales", "gross margin", 
        "fiscal year", "billion", "tax rate", "operations", "materially", "adversely", "assurance", "service", "tax", "iphone"
    }
    
    # Remove stopwords manually before tokenization
    words = text.split()
    words = [word for word in words if word.lower() not in total_stopwords]
    return " ".join(words)

def extract_topics_from_texts(extracted_sections, chunk_size = 128, overlap = 32, min_topic_size=5, custom_stopwords = None): 
    """
    Discover key topics dynamically from 10-K sections using BERTopic 
    - Uses machine learning to cluster similar words and find latent topics 
    - Does not rely on predefined categories 

    Parameters: 
    - extracted_sections (dict or list): 
        - if dict: keys are section names and values are text content 
        - if list: list of raw text documents 
        - if str: split into smaller chunks
    - min_topic_size (int): minimum number of documents per topic (default=5)

    Returns: 
    - pd.DataFrame: DataFrame containing topic details such as topic ID, words, relevance scores
    """
    if isinstance(extracted_sections, dict): 
        documents = [preprocess_text(doc) for section in extracted_sections.values() for doc in chunk_text(section, chunk_size, overlap)]
    elif isinstance(extracted_sections, list): 
        documents = [preprocess_text(doc) for text in extracted_sections for doc in chunk_text(text, chunk_size, overlap)]
    elif isinstance(extracted_sections, str):
        documents = [preprocess_text(doc) for doc in chunk_text(extracted_sections, chunk_size, overlap)]
    else: 
        raise TypeError("Error: Input must be a dictionary (section name -> text), a list, or a string.")


    #Check if GPU is available 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Load fast sentence transformer model for embeddings 
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    #Get default English Stopwords 
    default_stopwords = text.ENGLISH_STOP_WORDS

    #Define custom stopword list 
    if custom_stopwords is None: 
        total_stopwords = set(default_stopwords).union([
        "company", "companys", "business", "products", "services", "year", "2024", "2023", 
        "apple", "market", "product", "financial", "including", "form", "net sales", "gross margin", "fiscal year", "billion", "tax rate", "operations", "materially", "adversely"])

    else: 
        total_stopwords = set(default_stopwords).union(set(custom_stopwords))
    
    total_stopwords = list(total_stopwords)

    #Initialise vectorizer model to remove stopwords 
    vectorizer_model = CountVectorizer(
        stop_words=total_stopwords, 
        ngram_range=(1,2), 
        min_df=3, 
        token_pattern=r'\b[a-zA-Z]{3,}\b'
        )

    #Initialise and train BERTopic model 
    topic_model = BERTopic(
        embedding_model=embedding_model, 
        min_topic_size = min_topic_size, 
        vectorizer_model = vectorizer_model, 
        umap_model = PCA(n_components=10), #UMAP(n_neighbors=5, n_components=3, metric="cosine", low_memory=True) UMAP is computationally expensive, try to reduce number of neighbours and components 
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1, cluster_selection_epsilon=0.05, metric="euclidean", cluster_selection_method="leaf"), #HDBScan is an unsupervised clustering algorithm (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
        representation_model = MaximalMarginalRelevance(),
        verbose=True,
        )
    
    print(f"Number of documents: {len(documents)}")

    topics, prob  = topic_model.fit_transform(documents) #fit_transform() used to train the model and generate topics in one step. 
    #I.e. fit the BERTopic model to learn topic distributions, and Transforms the input documents into their associated topic assignments

    #Reduce Outliers 
    topics = topic_model.reduce_outliers(documents, topics)

    #Manually assign remaining outliers to the most probable topics 
    probs = topic_model.probabilities_ 
    outlier_docs = np.where(topic_model.topics_ ==-1)[0]

    for doc_id in outlier_docs: 
        best_topic = np.argmax(probs[doc_id]) #find the most probable topic
        topic_model.topics_[doc_id] = best_topic  #reassign the document 

    #Retrive topic information 
    topic_info = topic_model.get_topic_info() #BERTopic internally stores results as a Pandas DataFrame for easy analysis 

    return topic_info

#Summarisation 
#Load summarisation models once (avoid re-loading inside function to speed up)
abstract_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0, batch_size=1)
extractive_summarizer = LsaSummarizer() 

#Load NLP model for sentence processing 
nlp = spacy.load("en_core_web_sm")

def remove_redundant_sentences(summary): 
    """Uses fuzzy matching to remove repeated or redundant sentences"""
    doc = nlp(summary)
    unique_sentences= []
    seen = set()

    for sent in doc.sents: 
        clean_sent = sent.text.strip()
        if clean_sent.lower() not in seen: 
            seen.add(clean_sent.lower())
            unique_sentences.append(clean_sent)
    
    return " ".join(unique_sentences)


def summarize_text(text, summarization_type="extractive", max_length=250, min_length=50, overlap=100): 
    """"
    Summarises long text smoothly with different model types and smooth merging. 
    - Uses smart chunking for sections exceeding model limits. 

    Parameters: 
    - text (str): long text to be summarised 
    - summarization_type (str): "abstract" for generative (BART/T5), "extractive" for key sentences (DistilBERT)
    - max_length (int): Max length of each summary chunk 
    - min_length (int): Min length of each summary chunk
    - overlap (int): Overlap between chunks for smooth merging

    Returns: 
    - str: Final merged summary without redundant sentences

    """
    #Handling empty or short text
    if not text or len(text.split()) < min_length: 
        return "Error: Text too short for summarization"

    #Choose appropriate summarization model
    if summarization_type == "abstract":
        summarizer = abstract_summarizer
        needs_chunking = True #BART/T5 requires chunking
    elif summarization_type == "extractive":
        summarizer = extractive_summarizer 
        needs_chunking = False #Sumy can handle long text directly 
    else:
        raise ValueError("Error: summarization_type must be 'abstract' or 'extractive'.")

    #Chunk long text for processing
    text_chunks = chunk_text(text, chunk_size=512, overlap=overlap) if needs_chunking else [text]

    #Generate summaries 
    summaries = []
    for chunk in text_chunks: 
        if summarization_type == "abstract": 
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text'] #summarizer: pipeline from Hugging Face Transformers; takes text as input and returns list of dictionaries; hence extract first element at index [0], and each dictionary has key called "summary_text"
            print("Generated Summary:", summary)
        else: 
            parser = PlaintextParser.from_string(chunk, Tokenizer("english")) #Tokenizer splits text into sentences and words, then plaintext.parser.from_string() converts text into format that sumy can understand
            summary_sentences = summarizer(parser.document, sentences_count=10) #parser.document holds structured version of text, and summarizer takes structured text and extracts most important sentences as a list of sentence objects (sentence_count =3 means select top 3 most important sentences)
            summary = " ".join(str(sentence) for sentence in summary_sentences) if summary_sentences else chunk #converts list of objects into single string
    
        summaries.append(summary)
    
    #Merge and clean summary 
    final_summary = " ".join(summaries)
    final_summary = remove_redundant_sentences(final_summary)
  
    return final_summary

def analyze_sentiment(texts, method="nltk", batch_size=5): 
    """
    Perform sentiment analysis on a list of extracted text sections 
    Parameters: 
        - texts (list): List of text sections to analyze 
        - method (str): Choose between "transformers" (Hugging Face) or "nltk" (VADER). 
        - batch_size (int): Number of texts processed at a time (for memory efficiency)

    Returns: 
        - list: list of dictionaries containing text, sentiment label, and confidence score 
    """

    #Chunk text 
    chunks = chunk_text(texts) #breakdown texts into smaller chunks
    results = [] #empty list to store results 

    if method == "transformers": 
        #Load Hugging Face (transformer based) Sentiment Analysis pipeline
        sentiment_analyzer = pipeline("sentiment-analysis") #Deep learning powered, pre-trained on large corpora, slower, more opaque, but highly aware of context. Better used in long texts with nuanced language, outputs sentiment label + confidence score

        #Process in batches to avoid memory overload
        for i in range(0, len(chunks), batch_size): #iterate over chunks in steps of batch_size
            batch = chunks[i:i+batch_size]
            predictions = sentiment_analyzer(batch)
            for text, pred in zip(batch, predictions): 
                results.append({
                    "text": text, 
                    "sentiment": pred['label'], #Label is positive or negative
                    "score": pred['score'] #Confidence score (probability)
                })
    
    elif method == "nltk": 
        "Load VADER Sentiment Intensity Analyzer" #VADER (Valence Aware Dictionary and Sentiment Reasoner) -> lexicon and rule-based sentiment analysis tool. Pros: Simple, fast, works well on short text. Cons: may not be able to capture complex nuances that more powerful neural networks can
        sia = SentimentIntensityAnalyzer() #VADER works by having a lexicon (dictionary) of ~7.5k words, each associated with a valence score (-4 to +4). Then adjusts sentiment score based on heuristics (e.g. very good > good, negation, puncutation)
                                            #outputs 4 scores: pos: proportion of positive, neu: proportion of neutral, neg: proportion of negative, and compound: normalised overall score between -1 (most neg) and +1 most pos
        for chunk in chunks:                #generally anything between (-0.1 and 0.1) is neutral, above 0.1 positive and below -0.1 negative 
            scores = sia.polarity_scores(chunk)
            sentiment = "Positive" if scores['compound'] > 0 else "Negative" if scores['compound'] < 0 else "Neutral"
            results.append({
                "text": chunk, 
                "sentiment": sentiment, 
                "score": scores['compound']
            })
    
    else: 
        raise ValueError("Invalid method: Choose 'transformers' or 'nltk'.")
    
    #Create detailed results DataFrame 
    details_df = pd.DataFrame(results)

    #Compute summary 
    sentiment_counts = details_df["sentiment"].value_counts().to_dict()
    avg_score = details_df["score"].mean() if not details_df.empty else 0
    overall_sentiment = (
        "Positive" if avg_score >0 else 
        "Negative" if avg_score <0 else 
        "Neutral"
    )

    summary_data = {
        "overall_sentiment": overall_sentiment, 
        "average_score": avg_score, 
        "positive_chunks": sentiment_counts.get("Positive", 0),
        "negative_chunks": sentiment_counts.get("Negative", 0),
        "neutral_chunks": sentiment_counts.get("Neutral", 0)
    }

    summary_df = pd.DataFrame([summary_data])
    return summary_df, details_df #front-end and flask backend can only work with json serialisable objects (e.g. dictionaries). Dataframes are not, but can convert to returm .to_html format

    #Aggregate sentiment scores 
    #positive_count = sum(1 for r in results if r['sentiment'] == "Positive")
    #negative_count = sum(1 for r in results if r['sentiment'] == "Negative")
    #neutral_count = sum(1 for r in results if r['sentiment'] == "Neutral")

    #avg_score = sum(r["score"] for r in results) / len(results) if results else 0 
    #overall_sentiment = "Positive" if avg_score >0 else "Negative" if avg_score < 0 else "Neutral"

    #return {
    #    "overall_sentiment": overall_sentiment, 
    #    "average_score": avg_score, 
    #    "positive_chunks": positive_count,
    #    "negative_chunks": negative_count,
    #    "neutral_chunks": neutral_count, 
    #    "detailed_results": results,
    #}

def generate_wordcloud(word_freq_list, output_path="static/wordcloud.png", width=800, height=400, background_color="white"): 
    """
    Generates and saves a word cloud image from a list of (word, frequency) tuples. 
    Parameters: 
    - word_freq_list (list of tuples):[("word1", freq1), ("word2", freq2), ...]
    - output_path (str): where to save the word cloud image
    - width (int): width of the image
    - height (int): height of the image 
    - background_color (str): color of background
    """

    #Convert list of tuples to a dict 
    try: 
        freq_dict = {}

        #Case 1: Dataframe with columns [word, frequency]
        if isinstance(word_freq_list, pd.DataFrame): 
            if word_freq_list.shape[1] !=2: 
                raise ValueError("DataFrame must have exactly 2 columns (word, frequency).")
            freq_dict = dict(zip(word_freq_list.iloc[:,0], word_freq_list.iloc[:,1]))
        
        #Case 2: Dictionary: 
        elif isinstance(word_freq_list, dict): 
            freq_dict = word_freq_list

        #Case 3: List of tuples 
        elif isinstance(word_freq_list, list): 
            for item in word_freq_list: 
                if not isinstance(item, tuple) or len(item) !=2: 
                    raise ValueError(f"Each item must be a tuple (word, freq). Got: {item}")
                word, freq = item 
                if not isinstance(word, str) or not isinstance(freq, (int, float)): 
                    raise ValueError(f"Tuple must be (str, int/float). Got: ({type(word)}, {type(freq)})")
                freq_dict[word] = freq
        
        else: 
            raise TypeError("Unsupported input type. Must be list of tuples, dict, or DataFrame.")
        wordcloud = WordCloud(width=width, height=height, background_color=background_color).generate_from_frequencies(freq_dict)

        #Save to file 
        wordcloud.to_file(output_path)
        print(f"Word cloud saved to {output_path}")
        
    except Exception as e: 
        print(f"Error generating word cloud: {e}")


#Saving functions 
def save_summary_to_file(summary_text, output_path="static/summary.txt"): 
    with open(output_path, "w", encoding="utf-8") as f: 
        f.write(summary_text)
    print(f"Summary saved to: {output_path}")

def save_topics_to_csv(topic_df, output_path="static/topics.csv"):
    topic_df.to_csv(output_path, index=False)
    print(f"📈 Topics saved to: {output_path}")

import json

def save_sentiment_to_json(sentiment_dict, output_path="static/sentiment.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sentiment_dict, f, indent=2)
    print(f"💬 Sentiment saved to: {output_path}")

if __name__ == "__main__":
    #Sample test: load sample section and summarize it 
    from section_extractor import get_latest_10k_file, extract_text_from_10k, remove_toc, detect_section_headers, extract_main_sections
    
    file_path = get_latest_10k_file()
    if not file_path:
        print("No 10K file found")
        sys.exit(1)

    print(f"Using: {file_path}")
    raw_text = extract_text_from_10k(file_path)
    headers = detect_section_headers(raw_text)
    cleaned = remove_toc(raw_text, headers)
    extracted_sections = extract_main_sections(cleaned)

    #Define sections to retrieve 
    focus_sections = ["Item 1", "Item 7"]
    corpus = retrieve_corpus(extracted_sections, focus_sections)

    #Wordcloud generation and text processing
    top_words, tokens = process_text(corpus, top_n=30) 
    generate_wordcloud(top_words)

    #Extract key topics 
    try: 
        print("\n Extracting Topics...")
        topics_df = extract_topics_from_texts(corpus, chunk_size=128, overlap=32)
        save_topics_to_csv(topics_df)
        print(topics_df.head(5))
    except Exception as e: 
        print(f"Error in topic modelling: {e}")
    
    #Summarisation  
    try: 
        print("\n Generating Summary...")
        summary = summarize_text(corpus, summarization_type="extractive")
        save_summary_to_file(summary)
        print("Generated Summary", summary)

    except Exception as e: 
        print("Error in summarization:", str(e))
    
    #Extract sentiments
    try: 
        print("\n Analysing sentiments...")
        summary_df, details_df = analyze_sentiment(corpus)
        print(summary_df.head()), details_df 
        summary_df.to_csv("sentiment_summary.csv", index=False)
        details_df.to_csv("sentiment_details.csv", index=False)

        #save_sentiment_to_json(sentiment)
        #print("Overall Sentiment:", sentiment["overall_sentiment"])
        #print("Avg Score:", sentiment["average_score"])
        #print("👍 Pos:", sentiment["positive_chunks"], "👎 Neg:", sentiment["negative_chunks"], "😐 Neu:", sentiment["neutral_chunks"])
    except Exception as e:
        print("❌ Error in sentiment analysis:", e)