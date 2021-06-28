from gen.constants import *
from transformers import BertConfig, BertModel, BertTokenizer
from transformers import MobileBertConfig, MobileBertModel, MobileBertTokenizer
from transformers import AlbertConfig, AlbertModel, AlbertTokenizer
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

ckpt = {
        "bert": 'bert-base-uncased',
        "mobilebert": 'google/mobilebert-uncased',
        "albert": 'albert-base-v2',
        "roberta": 'roberta-base',
        }

def bert_config(args):
    if not args.use_bert:
        return
    bert_config = {
        "bert": BertConfig,
        "mobilebert": BertConfig,
        "albert": BertConfig,
        "roberta": RobertaConfig
        }[args.bert_model]
    bert_config = bert_config().from_pretrained(ckpt[args.bert_model])
    args.hidden_dim = bert_config.hidden_size
    args.enc_layer_num = bert_config.num_hidden_layers
    if hasattr(bert_config, 'embedding_size'):
        args.emb_dim = bert_config.embedding_size
    else:
        args.emb_dim = bert_config.hidden_size



def get_bert_model(args):
    model =  {
        "bert": BertModel,
        "mobilebert": MobileBertModel,
        "albert": AlbertModel,
        "roberta": RobertaModel,
        }[args.bert_model]
    return model.from_pretrained(ckpt[args.bert_model])


def get_bert_tknz(args):
    tknz = {
        "bert": BertTokenizer,
        "mobilebert": MobileBertTokenizer,
        "albert": AlbertTokenizer,
        "roberta": RobertaTokenizer,
        }[args.bert_model]
    return tknz.from_pretrained(ckpt[args.bert_model])


def mmt_word_ids_to_bert_ids(mmt_tokenized_seq, mmt_vocab, bert_tknz):
    str_sent = ''.join(mmt_vocab.seq_decode(mmt_tokenized_seq)).replace('  ', ' ')
    return bert_tknz.convert_tokens_to_ids(bert_tknz.tokenize(str_sent))
