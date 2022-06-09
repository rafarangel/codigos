# basic imports
import time
import pandas as pd
import tqdm
import os
import csv 

# for preprocessing
import re
from pandarallel import pandarallel

# for similarity model
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer

# configuration
tqdm.tqdm.pandas()
pandarallel.initialize()
pd.options.mode.chained_assignment = None

def lowercase(text): 
    text = str(text)
    text = text.lower()
    return text

def remove_punction(text): 
    text = str(text) 
    no_punct = re.sub(r'[!"#$%&\'()*+,-./<:>;=?@[/\/\]^_`{|}~]', ' ', text)
    return no_punct

def remove_space(text): # remove useless white spaces
    text = str(text)
    text = ' '.join(text.split())        
    return text

def preprocess(text):   # run all preprocessing functions
    text = lowercase(text)
    text = remove_punction(text)
    text = remove_space(text)
    return text

def encode_tfidf(phrase, vectorizer):
        # function to encode a sentence, it receive a string and return a numpy.array
        return vectorizer.transform(phrase).toarray().reshape(-1, 1)

def distance_cosine(vec):
    # calculates the cosine distance between two vectors
    return round(100*(1 - distance.cosine(vec[0], vec[1])), 2)

def compare_sentences(data):
    '''
    compares the sentence i with the sentences i+1, i+2, ..., i+n:
    An example, consider the sentences A, B, C, D, so
    A B
    A C
    A D
    B C
    B D
    C D
    '''
    
    list_data = []
    for i in tqdm.tqdm(data.index):
        for j in data.index[i+1:]:
            if data.loc[i, 'name'] != data.loc[j, 'name']:
                new_row = (data.loc[i, 'name'], data.loc[j, 'name'], data.loc[i, 'text'], data.loc[j, 'text'])
                list_data.append(new_row)
    return pd.DataFrame(list_data, columns = ['intent1', 'intent2', 'text1', 'text2'])
    
def load_stopword(text):
    global stopwords
    with open(text, 'r') as file:
        stopwords = [line.strip() for line in file] 
load_stopword('stopwords.txt')

def similar_words(data):
    comparation = data[['text1_cleaned', 'text2_cleaned']].values
    final = []
    for sentences in comparation:
        dual = []
        for word in sentences[0].split():
            if word in sentences[1].split() and word not in stopwords: dual.append(word)
        final.append(dual)
    return pd.Series(map(lambda x: ', '.join(x), final))

def prepare_data(data): # create two new columns with cleaned messages
    data['text1_cleaned'] = data['text1'].progress_apply(preprocess)
    data['text2_cleaned'] = data['text2'].progress_apply(preprocess)
    return data


def search_similarities(data, threshold = 0.7, save_dir = ''):
    
    macro_ini = time.time()
    
    # reads the data and prepares it
    
    data = compare_sentences(data)
    data = prepare_data(data)
    
    
    # Creating the vectorizer with respective corpus
    CORPUS = pd.concat([data.text1_cleaned, data.text2_cleaned]).drop_duplicates()
    tfidf_vectorizer = TfidfVectorizer() 
    tfidf_vectorizer.fit_transform(CORPUS)
    
    
    '''
    Creates a dict that contains the sentence and their respective encode:
    {'eu quero cartão': [0.3, 0, ..., 0.2], 'paguei a fatura': [0, 0.019, ..., 0.0]}
    '''
    dic_data = {}
    for phrase in tqdm.tqdm(CORPUS):
        dic_data[phrase] = encode_tfidf([phrase], tfidf_vectorizer)
    
    
    ''' 
    Creating a pandas.Series with pairs of encoded sentences:
    [[0.1, 0.321, ..., 0.032], [0.00, 0.00, ..., 0.01]]
    [[0.0, 0.053, ..., 0.200], [0.20, 0.12, ..., 0.13]]
    [[0.0, 0.150, ..., 0.153], [0.15, 0.07, ..., 0.00]]
    '''
    list_data = []
    for i in tqdm.tqdm(data.index):
        list_data.append((dic_data[data.loc[i, 'text1_cleaned']], dic_data[data.loc[i, 'text2_cleaned']]))
    list_data = pd.Series(list_data)
    
    
    
    # create a column named 'similarity' and calculates the cosine distance of each pair with multiprocessing
    ini = time.time()
    data['similarity'] = list_data.parallel_apply(distance_cosine)
    print('\n\nSimilarity model time:', round(time.time() - ini, 2), 'seconds')
    
    
    
    data['similar_words'] = similar_words(data)
    
    
    
    drop_columns = ['text1_cleaned', 'text2_cleaned']
    data = data.drop(drop_columns, axis = 1).sort_values('similarity', ascending = False)
    
    
    
    data = data[data.similarity > threshold]
    data.index = range(len(data))
    
    
    
    if save_dir:
        data.to_csv(save_dir, index = None, sep = ';')
    
    
    print('Full process time:', round(time.time() - macro_ini, 2), 'seconds')
    
    return data

def read_data_ALTU_export(dire):
  
    with open(dire, 'r') as f:
        reader = csv.reader(f)
        intents = list(reader)
        data = pd.DataFrame(list(map(lambda x: x[0].split(';'), intents)))
        
    phrases = data.iloc[:, 1:].apply(list, axis = 1)
    phrases.index = data.iloc[:, 0]
    phrases = phrases.apply(lambda x: pd.Series(x).dropna().tolist())

    lista = []
    for inte in phrases.index:
        for frase in phrases.loc[inte]:
            lista.append((inte, frase))
            
    data_root = pd.DataFrame(lista, columns = ['name', 'text'])
    data_root = data_root[data_root.text != '']
    data_root.text = data_root.text.apply(lambda x: x.replace('@', ''))

    return data_root
    

def read_data_xlsx(dire):
    try:
        df = pd.read_excel(dire, header = 1)[['Intenção', 'Frases de treinamento']]
    except:
        df = pd.read_excel(dire)[['Intenção', 'Frases de treinamento']]
        
    df = df.dropna(axis = 0)
    df = df[~df['Intenção'].isna()]
    frases = df['Frases de treinamento'].apply(lambda x:x.split('\n'))
    intencoes = df['Intenção'].apply(lambda x: x.replace('#', ''))
    frases.index = intencoes
    frases = frases.apply(lambda x: map(lambda y: ' '.join(y.split()), x)).apply(list)

    lista = []
    for inte in intencoes:
        for frase in frases.loc[inte]:
            lista.append((inte, frase))
            
    data_root = pd.DataFrame(lista, columns = ['name', 'text'])
    data_root = data_root[data_root.text != '']

    return data_root

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    dire = input('Digite o diretório do arquivo: ')
    thr = float(input('Digite o threshold do arquivo: '))


    try:
      data_root = read_data_ALTU_export(dire)
    except:
      data_root = read_data_xlsx(dire)

    data = search_similarities(data_root, thr)
    data.to_csv('similaridade_' + dire.split('/')[-1].split('.')[0] + '.csv', index = None, sep = '\t')

    return data
