import pickle, logging
import numpy as np

import revtok   # tokenizer

import torch
import torch.nn as nn
from gen.constants import *

class WordEmbedding(torch.nn.Module):
    '''
    inputs: x:          batch x ...
    outputs:embedding:  batch x ... x emb
            mask:       batch x ...
    '''

    def __init__(self, embedding_size, vocab, trainable=True, init_method='default',
                        scale=0.1, pretrain_path=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.vocab = vocab
        self.vocab_size = vocab.vocab_size
        self.init_method = init_method
        self.trainable = trainable
        self.scale = scale
        if 'pretrained' not in init_method:
            self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.init_weights(init_method, scale, pretrain_path)

    def init_weights(self, init_method, scale, pretrain_path):
        if init_method == 'default':
            return
        elif init_method == 'uniform':
            nn.init.uniform_(self.embedding.weight, a=-scale, b=scale)
        elif init_method == 'normal':
            nn.init.normal_(self.embedding.weight, std=scale)
        elif init_method == 'xavier':
            nn.init.xavier_uniform_(self.embedding.weight, gain=1.0)
        elif init_method == 'pretrained_glove':
            weight = init_form_glove(pretrain_path, self.vocab)
            self.embedding = nn.Embedding.from_pretrained(weight)
        else:
            raise NotImplementedError('Invalid initialization method: %s'%init_method)

        if not self.trainable:
            self.embedding.weight.requires_grad = False


    def forward(self, x):
        return self.embedding(x)  # batch x time x emb


class SpecialEmbedding(torch.nn.Module):
    '''
    '''
    def __init__(self, args, word_embedding, lang_vocab, action_vocab, bert_tknz=None):
        super().__init__()
        self.args = args
        self.word_embedding = word_embedding   # word embeddings
        self.lang_vocab = lang_vocab   # words in language directives
        self.action_vocab = action_vocab   #special object/action names
        self.bert_tknz = bert_tknz

        self.use_weights_in_word_embeddings()


    def use_weights_in_word_embeddings(self):
        weight = []
        if not self.args.use_bert:
            # an action token can be splited to up to 3 words
            func1 = lambda x: self.lang_vocab.seq_encode(revtok.tokenize(x), 3)
            func2 = lambda x: self.lang_vocab.seq_encode([x], 3)
        else:
            func1 = func2 =  lambda x: self.bert_tknz(x, add_special_tokens=False,
                padding='max_length', max_length=6)['input_ids']
        for idx, a in enumerate(self.action_vocab.words()):
            if a in ACTION_TO_WORDS:
                # print(words)
                wid_seq = func1(ACTION_TO_WORDS[a])
            elif a in OBJECTS_TO_WORDS:
                wid_seq = func1(OBJECTS_TO_WORDS[a])
            else:
                if self.args.use_bert:
                    a = 'start planning' if a == '[SOS]' else a
                    a = 'none' if a == 'None' else a
                wid_seq = func2(a)

            weight.append(wid_seq)

        # self.action_to_words = nn.Parameter(torch.as_tensor(weight), requires_grad=False)
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        self.action_to_words = torch.as_tensor(weight).to(device)


    def forward(self, x):
        # x: batch x seq_len
        to_word = self.action_to_words[x]   #batch x seq_len x 3
        embs = self.word_embedding(to_word)   #batch x seq_len x 3 x emb_dim
        return embs.sum(-2)   #batch x seq_len x emb_dim

        # embs = embs.sum(-2)
        # embs[x==2, :] = self.emb_fix(self.sos)   # embedding for  [SOS]
        # embs[x==126, :] = self.emb_fix(self.none)   # embedding for None
        # return embs


def init_form_glove(glove_path, vocab):
    """
    return a glove embedding matrix
    :param self:
    :param glove_file:
    :param initial_embedding_np:
    :return: np array of [V,E]
    """
    with open(glove_path, 'rb') as f:
        glove = pickle.load(f)
        glove_w2v = glove['data']
        mean = glove['vec_mean']
        std = glove['vec_std']

    E, V = 300, vocab.vocab_size
    weight = torch.zeros((V, E)).normal_(mean, std)
    count = 0
    for wid in range(V):
        word = vocab.id2w(wid).strip()
        if word in glove_w2v:
            glove_vec = glove_w2v[word]
            weight[wid] = torch.from_numpy(glove_vec)
            count += 1
    print('%d/%d words are intialized using pretrained GloVe embedding'%(count, V))
    return weight