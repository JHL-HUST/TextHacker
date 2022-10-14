from random import random
import re
import os
import pickle
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

import inflect
from autocorrect import Speller
import numpy as np
import torch
import random
import sys


replacement_patterns = [
    ("won\'t", "will not"),
    ("won\’t", "will not"),
    ("can\'t", "cannot"),
    ("can\’t", "cannot"),
    ("i\'m", "i am"),
    ("i\’m", "i am"),
    ("ain\'t", "is not"),
    ("ain\’t", "is not"),
    ("(\w+)\'ll", "\g<1> will"),
    ("(\w+)\’ll", "\g<1> will"),
    ("(\w+)n\'t", "\g<1> not"),
    ("(\w+)n’t", "\g<1> not"),
    ("(\w+)\'ve", "\g<1> have"),
    ("(\w+)\’ve", "\g<1> have"),
    ("(\w+)\'s", "\g<1> is"),
    ("(\w+)\’s", "\g<1> is"),
    ("(\w+)\'re", "\g<1> are"),
    ("(\w+)\’re", "\g<1> are"),
    ("(\w+)\'d", "\g<1> would"),
    ("(\w+)\’d", "\g<1> would")
]
def get_pos(sent, tagset='universal'):
    '''
    :param sent: list of word strings
    tagset: {'universal', 'default'}
    :return: list of pos tags.
    Universal (Coarse) Pos tags has  12 categories
        - NOUN (nouns)
        - VERB (verbs)
        - ADJ (adjectives)
        - ADV (adverbs)
        - PRON (pronouns)
        - DET (determiners and articles)
        - ADP (prepositions and postpositions)
        - NUM (numerals)
        - CONJ (conjunctions)
        - PRT (particles)
        - . (punctuation marks)
        - X (a catch-all for other categories such as abbreviations or foreign words)
    '''
    if tagset == 'default':
        word_n_pos_list = nltk.pos_tag(sent)
    elif tagset == 'universal':
        word_n_pos_list = nltk.pos_tag(sent, tagset=tagset)
    _, pos_list = zip(*word_n_pos_list)
    return pos_list

    
def get_stopwords():
    '''
    :return: a set of 266 stop words from nltk. eg. {'someone', 'anyhow', 'almost', 'none', 'mostly', 'around', 'being', 'fifteen', 'moreover', 'whoever', 'further', 'not', 'side', 'keep', 'does', 'regarding', 'until', 'across', 'during', 'nothing', 'of', 'we', 'eleven', 'say', 'between', 'upon', 'whole', 'in', 'nowhere', 'show', 'forty', 'hers', 'may', 'who', 'onto', 'amount', 'you', 'yours', 'his', 'than', 'it', 'last', 'up', 'ca', 'should', 'hereafter', 'others', 'would', 'an', 'all', 'if', 'otherwise', 'somehow', 'due', 'my', 'as', 'since', 'they', 'therein', 'together', 'hereupon', 'go', 'throughout', 'well', 'first', 'thence', 'yet', 'were', 'neither', 'too', 'whether', 'call', 'a', 'without', 'anyway', 'me', 'made', 'the', 'whom', 'but', 'and', 'nor', 'although', 'nine', 'whose', 'becomes', 'everywhere', 'front', 'thereby', 'both', 'will', 'move', 'every', 'whence', 'used', 'therefore', 'anyone', 'into', 'meanwhile', 'perhaps', 'became', 'same', 'something', 'very', 'where', 'besides', 'own', 'whereby', 'whither', 'quite', 'wherever', 'why', 'latter', 'down', 'she', 'sometimes', 'about', 'sometime', 'eight', 'ever', 'towards', 'however', 'noone', 'three', 'top', 'can', 'or', 'did', 'seemed', 'that', 'because', 'please', 'whereafter', 'mine', 'one', 'us', 'within', 'themselves', 'only', 'must', 'whereas', 'namely', 'really', 'yourselves', 'against', 'thus', 'thru', 'over', 'some', 'four', 'her', 'just', 'two', 'whenever', 'seeming', 'five', 'him', 'using', 'while', 'already', 'alone', 'been', 'done', 'is', 'our', 'rather', 'afterwards', 'for', 'back', 'third', 'himself', 'put', 'there', 'under', 'hereby', 'among', 'anywhere', 'at', 'twelve', 'was', 'more', 'doing', 'become', 'name', 'see', 'cannot', 'once', 'thereafter', 'ours', 'part', 'below', 'various', 'next', 'herein', 'also', 'above', 'beside', 'another', 'had', 'has', 'to', 'could', 'least', 'though', 'your', 'ten', 'many', 'other', 'from', 'get', 'which', 'with', 'latterly', 'now', 'never', 'most', 'so', 'yourself', 'amongst', 'whatever', 'whereupon', 'their', 'serious', 'make', 'seem', 'often', 'on', 'seems', 'any', 'hence', 'herself', 'myself', 'be', 'either', 'somewhere', 'before', 'twenty', 'here', 'beyond', 'this', 'else', 'nevertheless', 'its', 'he', 'except', 'when', 'again', 'thereupon', 'after', 'through', 'ourselves', 'along', 'former', 'give', 'enough', 'them', 'behind', 'itself', 'wherein', 'always', 'such', 'several', 'these', 'everyone', 'toward', 'have', 'nobody', 'elsewhere', 'empty', 'few', 'six', 'formerly', 'do', 'no', 'then', 'unless', 'what', 'how', 'even', 'i', 'indeed', 'still', 'might', 'off', 'those', 'via', 'fifty', 'each', 'out', 'less', 're', 'take', 'by', 'hundred', 'much', 'anything', 'becoming', 'am', 'everything', 'per', 'full', 'sixty', 'are', 'bottom', 'beforehand'}
    '''
    stop_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both',  'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except',  'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't",  'hasn', "hasn't",  'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly',  'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please','s', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow', 'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they','this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too','toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used',  've', 'was', 'wasn', "wasn't", 'we',  'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
    stop_words = set(stop_words)
    return stop_words

    
class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def rep(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s

def judge_pos(word):
    pos_tags = nltk.pos_tag([word])
    _, pos = pos_tags[0]
    return pos

def replace_with_synonym(word, synonym_pick_way, idx2word = None, word2idx = None, cos_sim = None):
    candidate_word_list = []
    candidate_value_list = []
    if synonym_pick_way == 'nltk':
        candidate_word_list = replace_with_synonym_nltk(word)
    elif synonym_pick_way == 'embedding':
        candidate_word_list, candidate_value_list = replace_with_synonym_embedding(word, idx2word, word2idx, cos_sim)
    else:
        assert None, "the way of picking synonym is wrong, please choose nltk or embedding"

    return candidate_word_list, candidate_value_list


def replace_with_synonym_embedding(word, idx2word, word2idx, cos_sim):
    synonym_words,synonym_values=[],[]
    if word in word2idx.keys():
        words_perturb_idx = word2idx[word]

        res = list(zip(*(cos_sim[words_perturb_idx])))

        for ii in res[1]:
            synonym_words.append(idx2word[ii])
        for ii in res[0]:
            synonym_values.append(ii)
    else:
        synonym_words.append(word)

    return synonym_words, synonym_values


def replace_with_synonym_nltk(word):
    pos_tags = nltk.pos_tag([word])
    _, pos = pos_tags[0]

    p = inflect.engine()
    correct = Speller()

    candidate_list = []
    word_list = []

    if pos in ['NN', 'NNS', 'NNP', 'NNPS']:
        word_set = wn.synsets(word, pos='n')
    elif pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', "MD"]:
        word_set = wn.synsets(word, pos='v')
    elif pos in ['RB', 'RBR', 'RBS']:
        word_set = wn.synsets(word, pos='r')
    elif pos in ['JJ', 'JJR', 'JJS']:
        word_set = wn.synsets(word, pos='a')
    else:
        word_set = wn.synsets(word)
    w_list = list(i._lemma_names for i in word_set)
    for w in w_list:
        word_list.extend(w)
    if word_list:
        word_set = set(word_list)
    if word_set:
        word_set.discard(word)
        if word_set:
            for candidate_word in word_set:
                if word.istitle():
                    nw = candidate_word.replace("_", " ").capitalize()
                else:
                    p_word = candidate_word.replace("_", " ")
                    if p_word.lower() == word.lower():
                        nw = word
                    else:
                        nw = p_word
                if pos in ["VBG"]:
                    nw = p.present_participle(nw)
                    nw = correct.autocorrect_word(nw)
                elif pos in ["NNS", "NNPS"]:
                    if not nw.endswith("s"):
                        nw = p.plural_noun(nw)
                        nw = correct.autocorrect_word(nw)
                candidate_list.append(nw)


    if word in candidate_list:
        candidate_list.remove(word)
    candidate_list.append(word)

    return candidate_list


def load_embedding_dict_info(embedding_path):
    idx2word = {}
    word2idx = {}

    with open(embedding_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    return idx2word, word2idx

# def load_cos_sim_matrix(cos_path):
#     cos_sim = []
#     if os.path.exists(cos_path):
#         with open(cos_path, "rb") as fp:
#             cos_sim = pickle.load(fp)
#     else:
#         if os.path.exists(embedding_path):
#             print('Start computing the cosine similarity matrix!')
#             embeddings = []
#             with open(embedding_path, 'r') as ifile:
#                 for line in ifile:
#                     embedding = [float(num) for num in line.strip().split()[1:]]
#                     embeddings.append(embedding)
#             embeddings = np.array(embeddings)
#             print(embeddings.T.shape)
#             norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
#             embeddings = np.asarray(embeddings / norm, "float64")
#             cos_sim = np.dot(embeddings, embeddings.T)
#             # pickle.dump([cos_sim], open(cos_path, 'wb'))
#         else:
#             assert None, "there is no embedding matrix and cos matrix"
#     return cos_sim


def load_embedding_npz(path):
    data = np.load(path)
    return [ w.decode('utf8') for w in data['words'] ], data['vals']

def load_embedding_txt(path):
    file_open = gzip.open if path.endswith(".gz") else open
    words = [ ]
    vals = [ ]
    with file_open(path, encoding='utf-8') as fin:
        fin.readline()
        for line in fin:
            line = line.rstrip()
            if line:
                parts = line.split(' ')
                words.append(parts[0])
                vals += [ float(x) for x in parts[1:] ]
    return words, np.asarray(vals).reshape(len(words),-1)

def load_embedding(path):
    if path.endswith(".npz"):
        return load_embedding_npz(path)
    else:
        return load_embedding_txt(path)


def pad(sequences, pad_token='<pad>', pad_left=True):
    ''' input sequences is a list of text sequence [[str]]
        pad each text sequence to the length of the longest
    '''
    max_len = max(5,max(len(seq) for seq in sequences))
    if pad_left:
        return [ [pad_token]*int(max_len-len(seq)) + list(seq) for seq in sequences ]
    return [ list(seq) + [pad_token]*(max_len-len(seq)) for seq in sequences ]



def create_one_batch(x, y, map2id, oov='<oov>'):
    oov_id = map2id[oov]
    x = pad(x)
    length = len(x[0])
    batch_size = len(x)
    x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
    x = torch.LongTensor(x)
    assert x.size(0) == length*batch_size
    return x.view(batch_size, length).t().contiguous().cuda(), torch.LongTensor(y).cuda()


def create_one_batch_x(x, map2id, oov='<oov>'):
    oov_id = map2id[oov]
    x = pad(x)
    length = len(x[0])
    batch_size = len(x)
    x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
    x = torch.LongTensor(x)
    assert x.size(0) == length*batch_size
    return x.view(batch_size, length).t().contiguous().cuda()


def create_batches(x, y, batch_size, map2id, perm=None, sort=False):

    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))

    x = [ x[i] for i in lst ]
    y = [ y[i] for i in lst ]

    sum_len = 0.
    for ii in x:
        sum_len += len(ii)
    batches_x = [ ]
    batches_y = [ ]
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx, by = create_one_batch(x[i*size:(i+1)*size], y[i*size:(i+1)*size], map2id)
        batches_x.append(bx)
        batches_y.append(by)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]
        batches_y = [ batches_y[i] for i in perm ]

    sys.stdout.write("{} batches, avg sent len: {:.1f}\n".format(
        nbatch, sum_len/len(x)
    ))

    return batches_x, batches_y


# shuffle training examples and create mini-batches
def create_batches_x(x, batch_size, map2id, perm=None, sort=False):

    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))

    x = [ x[i] for i in lst ]

    sum_len = 0.0
    batches_x = [ ]
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx = create_one_batch_x(x[i*size:(i+1)*size], map2id)
        sum_len += len(bx)
        batches_x.append(bx)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]

    # sys.stdout.write("{} batches, avg len: {:.1f}\n".format(
    #     nbatch, sum_len/nbatch
    # ))

    return batches_x