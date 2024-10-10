import pyterrier as pt
if not pt.started():
  pt.init()

# -*- coding: utf-8 -*-
import string
import pandas as pd
import re
import globals
import arabicstopwords.arabicstopwords as stp
from snowballstemmer import stemmer
from mContriever import MContriever



def rank_by_score(score, actual):
    # Check if the length of score array matches the number of rows in actual DataFrame
    if len(score) != len(actual):
        raise ValueError("Length of score array does not match the number of rows in actual DataFrame")

    # Combine score and actual into tuples for sorting
    combined = [(s, a) for s, a in zip(score, actual)]
    
    # Sort combined based on score in descending order
    combined.sort(reverse=True)
    
    # Extract actual array from sorted combined list
    ranked_actual = [item[1] for item in combined]
    
    return ranked_actual


def clean(text):
    text = re.sub(r"http\S+", " ", text)  # remove urls
    text = re.sub(r"@[\w]*", " ", text)  # remove handles
    text = re.sub(r"[\.\,\#_\|\:\?\?\/\=]", " ", text) # remove special characters
    text = re.sub(r"\t", " ", text)  # remove tabs
    text = re.sub(r"\n", " ", text)  # remove line jump
    text = re.sub(r"\s+", " ", text)  # remove extra white space

    # text = remove_harkat(text)
    text = text.strip()
    return text



def remove_punctuations_tashkeel(text):
    """
    The input should be arabic string
    """
    punctuations = """`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ""" + string.punctuation

    arabic_diacritics = re.compile(
        """
                                ّ    | # Shadda
                                َ    | # Fatha
                                ً    | # Tanwin Fath
                                ُ    | # Damma
                                ٌ    | # Tanwin Damm
                                ِ    | # Kasra
                                ٍ    | # Tanwin Kasr
                                ْ    | # Sukun
                                ـ     # Tatwil/Kashida
                         """,
        re.VERBOSE,
    )

    # remove_punctuations
    translator = str.maketrans("", "", punctuations)
    text = text.translate(translator)

    # remove Tashkeel
    text = re.sub(arabic_diacritics, "", text)

    return text


def remove_longation(text):
    # remove longation
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_harakaat(text):
    # harakaat and tatweel (kashida) to remove
    accents = re.compile(r"[\u064b-\u0652\u0640]")

    # Keep only Arabic letters/do not remove number
    arabic_punc = re.compile(r"[\u0621-\u063A\u0641-\u064A\d+]+")
    text = " ".join(arabic_punc.findall(accents.sub("", text)))
    text = text.strip()
    return text


def normalize(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    return text


def remove_punctuation(text):
    # Removing punctuations in string using regex
    text = re.sub(r'[^\w\s]', '', text)
    return text

#removing Stop Words function
def remove_stopWords(sentence):
    terms=[]
    stopWords= set(stp.stopwords_list())
    for term in sentence.split() :
        if term not in stopWords :
           terms.append(term)
    return " ".join(terms)

def clean_and_normalized_text(text):
    # text = text.replace('"', ' ')  # remove double quotes
    text = normalize(text)
    text = clean(text)
    return text

ar_stemmer = stemmer("arabic")
#define the stemming function
def stem(sentence):
    return " ".join([ar_stemmer.stemWord(i) for i in sentence.split()])

def data_preprocessing_text(text):
    if pd.notna(text):  # Check if the value is not NaN
        # text = remove_emoji(text)
        text = remove_punctuations_tashkeel(text)
        text = remove_longation(text)
        text = remove_harakaat(text)
        text = remove_stopWords(text)
        text = clean_and_normalized_text(text)
        text = stem(text)
        return text
    else:
        return ""
    

input_file_path = "data.csv"

df = pd.read_csv(input_file_path, encoding='utf-8-sig')

df['Question'] = df['Question'].apply(data_preprocessing_text)
df['Answer'] = df['Answer'].apply(data_preprocessing_text)
df['Category'] = df['Category'].apply(data_preprocessing_text)
df['Title'] = df['Title'].apply(data_preprocessing_text)

df['docno'] = range(1, len(df) + 1)
df['docno'] = df['docno'].astype(str)

df.to_csv(input_file_path, index=False, encoding='utf-8-sig')

indexer = pt.DFIndexer("./drive/MyDrive/arabic_collec_index", overwrite=True)
indexer.setProperty("tokeniser", "UTFTokeniser")

index_ref = indexer.index(df['Answer'], df['Question'],  df['docno'])


index = pt.IndexFactory.of(index_ref)

DEPTH = 5
def initial_rank(query):
  bm25_retr = pt.BatchRetrieve(index, controls = {"wmodel": "BM25"},num_results=DEPTH)
  preprocessed_query = data_preprocessing_text(query)
  results = bm25_retr.transform([preprocessed_query])
  return results


import pandas as pd

eval_df = pd.read_csv("data/IslamwebEvaluation.csv")

queries = eval_df['Queries'].tolist()
qid = eval_df['QID'].tolist()

# Create a DataFrame to store queries and their IDs
queriesDF = pd.DataFrame({
    'qid': qid,
    'raw_query': queries
})

queriesDF['query'] = queriesDF['raw_query'].apply(data_preprocessing_text)

bm25_retr = pt.BatchRetrieve(index_ref, controls = {"wmodel": "BM25"},num_results=5)
###initial retreival 
bm25_res=bm25_retr.transform(queriesDF)

print(bm25_res)

### re-ranking
queries = bm25_res["query"]
docnos = bm25_res["docno"]
answer_after_rerank = pd.DataFrame(columns=["qid", "docno", "rank" ,"score", "query"])
 
for doc in range(len(bm25_res)):
    score = [] 
    answers_per_query = pd.DataFrame(columns=["qid", "docno", "rank" ,"score", "query"])

    for a in range(5):### do that for the retrieved 5 answers
        #emb for query 
        query_emb = MContriever(queries[doc])
        results = query_emb.get_embeddings()
        query_embeddings = results['mcontriever_embeddings']

        #emb for ans
        d = docnos[doc+a]
        answer = df.loc[df['docno'] == d, 'Answer'].values[0]
        answer_emb = MContriever(answer)
        results = query_emb.get_embeddings()
        answer_embeddings = results['mcontriever_embeddings']
        #append the scores of all ansers
        score.append(query_embeddings[0] @ answer_embeddings[0])
        ###append the answers to new dataframe object
        answers_per_query.append(bm25_res.iloc[doc], ignore_index=True)
    ##do reranking 
    answers_per_query_after_rerank = rank_by_score(score, answers_per_query)
    ###append the new ranked answers
    answer_after_rerank.append(answers_per_query_after_rerank, ignore_index=True)







qrels = pd.read_csv("data/qrels.txt", sep=',', names=['qid', 'docno', 'Q0', 'label'], header=0)
qrels['docno'] = qrels['docno'].astype(str)

bm25_eval = pt.Utils.evaluate(answer_after_rerank,qrels[['qid','docno','label']],metrics=["map","recip_rank","P", "recall_1", "recall_2", "recall_3", "recall_4", "recall_5"])
print(bm25_eval)