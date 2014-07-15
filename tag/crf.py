#!/usr/bin/env python

'''
Part-of-Speech tagger with CRFsuite

Author: Yi Yang
Email: yangyiycc@gmail.com
'''

from __future__ import division
import re, crfsuite
import numpy as np
from sys import argv
from scipy.sparse import *
from scipy.sparse import identity
from scipy import transpose
from scipy.sparse.linalg import *


def struct_dropout(xx, pivots):
    lam = 1
    xfreq = xx[pivots,:]
    P = xfreq * transpose(xx)
    print('finish computing P')
    normvec = np.squeeze(np.asarray(xx.sum(1)))
    normvec = normvec.astype(float)
    d = len(normvec)
    for i in xrange(d): normvec[i] = 1 / (normvec[i] + lam)
    Q = spdiags([normvec],[0],d,d)
    print('finish computing Q')
    W = P * Q
    print('finish computing W')
    return transpose(csc_matrix(W * xx).tanh().todense())


def train_test(train_conll, test_conll, model):
    train_data = attribute_getter(train_conll)
    test_data = attribute_getter(test_conll)
    learn_representation(train_data, test_data, 200)
    train(train_data, model)
    test(test_data, model)


def learn_representation(train_data, test_data, pivot_num):
    feature_dict = {}
    data = []
    rows = []
    cols = []
    fea_idx = 0
    inst_idx = 0
    for xseq, yseq in train_data:
        for item in xseq:
            for attr in item:
                data.append(getattr(attr, 'value'))
                cols.append(inst_idx)
                feature = getattr(attr, 'attr')
                if feature in feature_dict:
                    rows.append(feature_dict[feature])
                else:
                    feature_dict[feature] = fea_idx
                    rows.append(fea_idx)
                    fea_idx = fea_idx + 1
            inst_idx = inst_idx + 1

    for xseq, yseq in test_data:
        for item in xseq:
            for attr in item:
                data.append(getattr(attr, 'value'))
                cols.append(inst_idx)
                feature = getattr(attr, 'attr')
                if feature in feature_dict:
                    rows.append(feature_dict[feature])
                else:
                    feature_dict[feature] = fea_idx
                    rows.append(fea_idx)
                    fea_idx = fea_idx + 1
            inst_idx = inst_idx + 1

    xx = csc_matrix((data,(rows,cols)))
    print('start compute pivots')
    pivots = compute_pivots(xx, len(train_data), pivot_num)
    print('start learn hx')
    hx = struct_dropout(xx, pivots)

    inst_idx = 0
    for xseq, yseq in train_data:
        for item in xseq:
            item = crfsuite.Item(item)
            for i in xrange(pivot_num):
                #print(hx[i, inst_idx]),
                item.append(crfsuite.Attribute('new='+str(i), hx[inst_idx, i]))
            inst_idx = inst_idx + 1
        #print(inst_idx)

    for xseq, yseq in test_data:
        for item in xseq:
            item = crfsuite.Item(item)
            for i in xrange(pivot_num):
                item.append(crfsuite.Attribute('new'+str(i), hx[inst_idx, i]))
            inst_idx = inst_idx + 1
        #print(inst_idx)

    print(hx.shape)
    print(inst_idx)


def compute_pivots(xx, offset, pivot_num):
    freq = np.squeeze(np.asarray(xx.sum(1)))
    return np.argsort(freq)[-pivot_num:]
#    train_freq = xx[:,:offset-1].sum(1)
#    test_freq = xx[:,offset:].sum(1)
#    train_idx = np.argsort(train_freq, 0)
#    test_idx = np.argsort(test_freq, 0)
#    top_idx = set()
#    idx = xx.shape[1]-1
#    pivots = []
#    while len(pivots) < pivot_num:
#        if train_idx[idx,0] in top_idx:
#            pivots.append(train_idx[idx,0])
#            print(train_freq[train_idx[idx,0],0])
#            print(test_freq[train_idx[idx,0],0])
#        else:
#            top_idx.add(train_idx[idx,0])
#        if test_idx[idx,0] in top_idx:
#            pivots.append(test_idx[idx,0])
#            print(train_freq[test_idx[idx,0],0])
#            print(test_freq[test_idx[idx,0],0])
#        else:
#            top_idx.add(test_idx[idx,0])
#        idx = idx - 1
#    return pivots



def train(data, model):
    trainer = crfsuite.Trainer()

    for xseq, yseq in data:
        trainer.append(xseq, yseq, 0)

    # Use L2-regularized SGD and 1st-order dyad features
    trainer.select('l2sgd', 'crf1d')

    # Set the coefficient for L2 regularization to 0.1
    trainer.set('c2', '0.1')
    trainer.set('max_iterations', '50')

    # This demonstrates how to list parameters and obtain their values.
    for name in trainer.params():
        print name, trainer.get(name), trainer.help(name)

    # Start training
    trainer.train(model, -1)


def test(data, model):
    # Create a tagger object.
    tagger = crfsuite.Tagger()
    
    # Load the model to the tagger.
    tagger.open(model)

    total = 0
    corr = 0
    for xseq, yseq in data:
        # Obtain the label sequence predicted by the tagger
        ypred = tagger.tag(xseq)
        for i in xrange(0, len(yseq)):
            total = total + 1
            if yseq[i] == ypred[i]:
                corr = corr + 1

    accuracy = corr / total
    print('accuracy = ' + str(accuracy) + ' (' + str(corr) + '/' + str(total) + ')')


def attribute_getter(infile):
    attrs = []
    tags = []
    tokens = []
    fin = open(infile, 'r')
    for line in fin:
        line = line.strip()
        if line == '':
            xseq = crfsuite.ItemSequence()
            yseq = crfsuite.StringList()
            for index in xrange(0, len(tokens)):
                features = feature_detector(tokens, index)
                # Append attributes to the item
                item = crfsuite.Item()
                for key in features:
                    item.append(crfsuite.Attribute(key + '=' + features[key]))
                # Append the item to the item sequence.
                xseq.append(item)
                yseq.append(tags[index])
            attrs.append((xseq, yseq))
            del tags[:], tokens[:]
        else:
            parts = line.split()
            tags.append(parts[1])
            tokens.append(parts[0])
    fin.close()
    return attrs


def feature_detector(tokens, index):
    word = tokens[index]
    if index == 0:
        prevword = '<s>'
        prevprevword = '<null>'
    elif index == 1:
        prevword = tokens[index-1]
        prevprevword = '<s>'
    else:
        prevword = tokens[index-1]
        prevprevword = tokens[index-2]
    if index == len(tokens)-1:
        nextword = '</s>'
        nextnextword = '<null>'
    elif index == len(tokens)-2:
        nextword = tokens[index+1]
        nextnextword = '</s>'
    else:
        nextword = tokens[index+1]
        nextnextword = tokens[index+2]

    if re.search('\d', word):
        containsNum = 'true'
    else:
        containsNum = 'false'

    if re.search('[A-Z]', word):
        containsUpper = 'true'
    else:
        containsUpper = 'false'

    if re.search('-', word):
        containsHyphen = 'true'
    else:
        containsHyphen = 'false'

#    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
#        shape = 'number'
#    elif re.match('\W+$', word):
#        shape = 'punct'
#    elif re.match('[A-Z][a-z]+$', word):
#        shape = 'upcase'
#    elif re.match('[a-z]+$', word):
#        shape = 'downcase'
#    elif re.match('\w+$', word):
#        shape = 'mixedcase'
#    else:
#        shape = 'other'

    features = {
        'word': word,
        'prevprevword': prevprevword,
        'prevword': prevword,
        'nextword': nextword,
        'nextnextword': nextnextword,
        'prefix1': word[:1],
        'prefix2': word[:2],
        'prefix3': word[:3],
        'prefix4': word[:4],
        'suffix4': word[-4:],
        'suffix3': word[-3:],
        'suffix2': word[-2:],
        'suffix1': word[-1:],
        'ContainsNum' : containsNum,
        'ContainsUpper' : containsUpper,
        'ContainsHyphen' : containsHyphen
#        'shape': shape
        }
    return features


if __name__ == '__main__':
    train_test(argv[1], argv[2], argv[3])
