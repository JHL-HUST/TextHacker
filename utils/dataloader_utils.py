import gzip
import os
import sys
import io
import re
import random
import csv
import numpy as np
import torch
import string


csv.field_size_limit(sys.maxsize)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def read_corpus(path, csvf=False , clean=True, MR=True, encoding='utf8', shuffle=False, lower=True):
    data = []
    labels = []
    if not csvf:
        with open(path, encoding=encoding) as fin:
            for line in fin:
                if MR:
                    label, sep, text = line.partition(' ')
                    label = int(label)
                else:
                    label, sep, text = line.partition(',')
                    label = int(label) - 1
                if clean:
                    text = clean_str(text.strip()) if clean else text.strip()
                if lower:
                    text = text.lower()
                labels.append(label)
                data.append(text.split())
    else:
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                text = line[0]
                label = int(line[1])
                if clean:
                    text = clean_str(text.strip()) if clean else text.strip()
                if lower:
                    text = text.lower()
                labels.append(label)
                data.append(text.split())

    if shuffle:
        perm = list(range(len(data)))
        random.shuffle(perm)
        data = [data[i] for i in perm]
        labels = [labels[i] for i in perm]

    return data, labels


def read_nli_data(filepath, target_model='infersent', lowercase=False, ignore_punctuation=False, stopwords=[]):
    """
    Read the premises, hypotheses and labels from some NLI dataset's
    file and return them in a dictionary. The file should be in the same
    form as SNLI's .txt files.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (and MultiNLI) dataset.

    Returns:
        A dictionary containing three lists, one for the premises, one for
        the hypotheses, and one for the labels in the input data.
    """
    if target_model == 'Bert_NLI':
        labeldict = {"contradiction": 0,
                      "entailment": 1,
                      "neutral": 2}
    else:
        labeldict = {"entailment": 0,
                     "neutral": 1,
                     "contradiction": 2}
    with open(filepath, 'r', encoding='utf8') as input_data:
        premises, hypotheses, labels = [], [], []

        # Translation tables to remove punctuation from strings.
        punct_table = str.maketrans({key: ' ' for key in string.punctuation})

        for idx, line in enumerate(input_data):
            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue
            
            premise = line[1]
            hypothesis = line[2]

            if lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            # Each premise and hypothesis is split into a list of words.
            premises.append([w for w in premise.rstrip().split()
                             if w not in stopwords])
            hypotheses.append([w for w in hypothesis.rstrip().split()
                               if w not in stopwords])
            labels.append(labeldict[line[0]])

        return premises, hypotheses, labels