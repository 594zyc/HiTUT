import logging, json, os


class Vocab(object):
    def __init__(self, vocab_size=0, unk_idx=1, special_tokens=[]):
        self._idx2word = {}
        self._word2idx = {}
        self._freq_dict = {}
        self.unk_idx = unk_idx
        self.special_tokens = special_tokens
        for w in self.special_tokens:
            self._add_to_vocab(w)
        self.vocab_size = vocab_size if vocab_size > len(self) else len(self)


    def __len__(self):
        return len(self._idx2word)


    def __contains__(self, key):
        return key in self._word2idx


    def words(self):
        return list(self._word2idx.keys())


    def _add_to_vocab(self, word):
        if word not in self._word2idx:
            idx = len(self._idx2word)
            self._idx2word[idx] = word
            self._word2idx[word] = idx


    def add_word_counts(self, word):
        # word count add by 1 in the frequency dict
        if word not in self._freq_dict:
            self._freq_dict[word] = 0
        self._freq_dict[word] += 1


    def construct(self, vocab_path):
        # add #VOCAB_SIZE most frequently appeared words to the vocabulary
        self._freq_dict = dict(sorted(self._freq_dict.items(), key=lambda item: -item[1]))
        for word in self._freq_dict:
            self._add_to_vocab(word)
            if len(self._idx2word) >= self.vocab_size:
                break
        actual_vocab_size = len(self._idx2word)
        logging.info('Vocabulary constructed')
        logging.info('Set vocabulary size: %d'%self.vocab_size)
        logging.info('Actual vocabulary size: %d'%actual_vocab_size)
        self.save(vocab_path)
        logging.info('Vocabulary file saved to %s'%vocab_path)


    def load(self, vocab_path, vocab_size=None):
        self._word2idx = json.loads(open(vocab_path+'.w2id.json', 'r').read())
        self._idx2word = {}
        if os.path.exists(vocab_path+'.freq.json'):
            self._freq_dict = json.loads(open(vocab_path+'.freq.json', 'r').read())
        else:
            self._freq_dict = {}
        for w, idx in self._word2idx.items():
            self._idx2word[idx] = w
        actual_vocab_size = len(self._idx2word)
        if vocab_size is None:
            self.vocab_size = actual_vocab_size
            print('Vocab file loaded. Size: %d' % (actual_vocab_size))
        else:
            self.vocab_size = vocab_size
            for word, idx in list(self._word2idx.items()):
                if idx >= vocab_size:
                    del self._word2idx[word]
                    del self._idx2word[idx]
            print('Vocab file loaded. Size: %d' % (len(self)))


    def save(self, vocab_path):
        with open(vocab_path+'.w2id.json', 'w') as f:
            json.dump(self._word2idx, f, indent=2)
        if self._freq_dict:
            with open(vocab_path+'.freq.json', 'w') as f:
                json.dump(self._freq_dict, f, indent=2)


    def w2id(self, word):
        if word not in self._word2idx:
            return self.unk_idx
        else:
            return self._word2idx[word]


    def id2w(self, idx):
        if type(idx) is not int:
            idx = int(idx.item())
        return self._idx2word.get(idx, self.unk_idx)


    def seq_encode(self, word_list, pad_to_length=None):
        if pad_to_length is None or pad_to_length <= len(word_list):
            return [self.w2id(_) for _ in word_list]
        else:
            pad_num = pad_to_length - len(word_list)
            return [self.w2id(_) for _ in word_list] + [0] * pad_num


    def seq_decode(self, index_list):
        return [self.id2w(_) for _ in index_list]