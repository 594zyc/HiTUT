import os, sys, collections, time, logging
import numpy as np
import revtok   # tokenizer

import torch
from torch import nn
from torch.nn import functional as F



sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))

from models.nn.nn_utils import drop_input
from models.nn.transformer import Encoder
from models.nn.embedding import WordEmbedding, SpecialEmbedding
from models.utils.bert_utils import get_bert_model, mmt_word_ids_to_bert_ids
from gen.constants import *



class MultiModalTransformer(nn.Module):

    def __init__(self, args, dataset):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.vocab = dataset.vocab
        self.bert_tknz = dataset.bert_tknz if args.use_bert else None
        self.action_vocab = dataset.dec_in_vocab
        self.dec_vocabs = {
            'type_high': self.dataset.dec_out_vocab_high,
            'type_low': self.dataset.dec_out_vocab_low,
            'arg': self.dataset.dec_out_vocab_arg,
        }
        self.can_navi = args.train_level !='high' and args.low_data != 'mani'
        self.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        self.seq_len = self.args.max_enc_length
        self.emb_dim = args.emb_dim
        self.image_size = self.args.image_size
        self.vis_len = args.topk_objs + 1
        self.lang_len = args.lang_max_length
        self.his_len = args.history_max_length * 2
        assert self.seq_len == (1+self.vis_len+self.lang_len+self.his_len)
        # 0: pad, 1: vision, 2: language, 3: action history, 4: cls (optional, may be used
        # to obtain a state representation)
        self.modal_indicator = self.totensor([4]+[1]*self.vis_len+[2]*self.lang_len+[3]*self.his_len)
        # self.print_parameters()

        self.high_actions = self.totensor(self.action_vocab.seq_encode(HIGH_ACTIONS))
        self.low_actions = self.totensor(self.action_vocab.seq_encode(LOW_ACTIONS))
        self.objs = self.totensor(self.action_vocab.seq_encode(ACTION_ARGS))

        self.position_indexes = self.totensor(list(range(args.max_enc_length)))

        self.zero = self.totensor(0)

        self._construct_model()
        self.print_parameters()


    def totensor(self, x):
        return torch.tensor(x).to(device=self.device)

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %.2fM" % (amount/1e6))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %.2fM" % (amount/1e6))


    def _construct_model(self):
        args = self.args

        # input
        if not self.args.use_bert:
            # trainable positional embeddings
            self.positional_emb = nn.Embedding(self.seq_len, args.hidden_dim)
            # 0: pad, 1: vision, 2: language, 3: action history, 4: [CLS]
            self.modality_emb = nn.Embedding(5, args.hidden_dim)

            # the transformer encoder
            self.encoder = Encoder(n_layers=args.enc_layer_num, n_head=args.head_num,
                d_model=args.hidden_dim, d_inner=args.inner_dim, dropout=args.dropout)

            # word embedding for natural language
            self.word_emb = WordEmbedding(args.emb_dim, self.vocab, trainable=not args.emb_freeze,
                init_method=args.emb_init, pretrain_path='data/glove840B300d_extracted.pkl')

            # token embedding for action types and arguments (objects)
            self.action_emb = SpecialEmbedding(args, self.word_emb,
                self.vocab, self.action_vocab)
            # self.action_emb = nn.Embedding(len(self.action_vocab), args.emb_dim)

            # features -> transformer encoder input
            self.vision_to_enc_input = nn.Sequential(
                    nn.Linear(args.emb_dim + 7, args.hidden_dim),
                    nn.LayerNorm(args.hidden_dim, eps=1e-6)
                )
            self.lang_to_enc_input = nn.Sequential(
                    self.word_emb,
                    nn.Linear(args.emb_dim, args.hidden_dim),
                    nn.LayerNorm(args.hidden_dim, eps=1e-6)
                )
            self.action_to_enc_input = nn.Sequential(
                    self.action_emb,
                    nn.Linear(args.emb_dim, args.hidden_dim),
                    nn.LayerNorm(args.hidden_dim, eps=1e-6)
                )

            # historical visual features as additional input
            if self.args.enable_feat_vis_his:
                self.vis_his_to_enc_input = nn.Sequential(
                    nn.Linear(args.emb_dim + 7, args.hidden_dim),
                    nn.LayerNorm(args.hidden_dim, eps=1e-6)
                )

            # agent posture features as additional input
            if self.args.enable_feat_posture:
                self.rotation_emb = nn.Embedding(4, args.hidden_dim)
                self.horizon_emb = nn.Embedding(12, args.hidden_dim)
                self.pos_ln = nn.LayerNorm(args.hidden_dim, eps=1e-6)

        else:
            self.encoder = get_bert_model(args)
            self.word_emb = self.encoder.embeddings.word_embeddings
            scale =torch.std(self.word_emb.weight[:1000]).item()
            self.action_emb = SpecialEmbedding(args, self.word_emb,
                self.vocab, self.action_vocab, self.bert_tknz)
            self.vis_feat_to_emb = nn.Sequential(
                    nn.Linear(7, args.emb_dim),
                    # nn.LayerNorm(args.emb_dim, eps=1e-6)
                )
            nn.init.normal_(self.vis_feat_to_emb[0].weight, std=scale)
            nn.init.constant_(self.vis_feat_to_emb[0].bias, 0)

            self.arg_head = nn.Linear(args.hidden_dim, len(ACTION_ARGS))
            self.high_head = nn.Linear(args.hidden_dim, len(HIGH_ACTIONS))
            self.low_head = nn.Linear(args.hidden_dim, len(LOW_ACTIONS))
            if self.args.enable_feat_vis_his:
                self.vis_his_to_enc_input = nn.Linear(args.emb_dim + 7, args.emb_dim)
            if self.args.enable_feat_posture:
                self.rotation_emb = nn.Embedding(4, args.hidden_dim)
                self.horizon_emb = nn.Embedding(12, args.hidden_dim)
                self.pos_ln = nn.LayerNorm(args.hidden_dim, eps=1e-6)
                nn.init.normal_(self.rotation_emb.weight, std=scale)
                nn.init.normal_(self.horizon_emb.weight, std=scale)

        # output

        # transformatinos before going into the classification heads
        if args.hidden_dim != args.emb_dim:
            self.fc1 = nn.Linear(args.hidden_dim, args.emb_dim)
            self.fc2 = nn.Linear(args.hidden_dim, args.emb_dim)
            self.fc3 = nn.Linear(args.hidden_dim, args.emb_dim)
        else:
            self.fc1 = self.fc2 = self.fc3 = nn.Identity()

        # navigation status monitoring
        if self.can_navi and self.args.auxiliary_loss_navi:
            self.visible_monitor = nn.Linear(args.hidden_dim, 1)
            self.reached_monitor = nn.Linear(args.hidden_dim, 1)
            self.progress_monitor = nn.Linear(args.hidden_dim, 1)

        self.dropout = nn.Dropout(args.dropout)
        # self.softmax = nn.Softmax(dim=1)


    def forward(self, batch, is_optimizing=True):
        task_type = batch['batch_type'][0]
        inputs, enc_masks, labels = self.process_batch(batch)
        if not self.args.use_bert:
            enc_output, attns_list = self.encoder(inputs, enc_masks)
            # enc_output: batch x seq_len x hid
        else:
            outputs = self.encoder(inputs_embeds=inputs, attention_mask =enc_masks, output_attentions=True)
            enc_output = outputs.last_hidden_state
            attns_list = outputs.attentions

        type_logits, arg_logits, mask_logits, navi_logits = self.outputs_to_logits(enc_output,
            attns_list, labels['type_pos'], labels['arg_pos'], task_type)   #_bert

        if not is_optimizing:
            type_preds, arg_preds, mask_preds, navi_preds = self.pred(type_logits, arg_logits, mask_logits, navi_logits)
            return type_preds, arg_preds, mask_preds, navi_preds, labels
        type_loss, arg_loss, mask_loss, navi_loss = self.get_loss(type_logits, arg_logits, mask_logits, navi_logits,
            labels, task_type)
        return type_loss, arg_loss, mask_loss, navi_loss


    def process_batch(self, batch):
        """convert batch data to tensor matrix

        Args:
            batch: list of from items from AlfredPyTorchDataset

        Returns:
            inputs: encoder input matrix (tensor of size [batch x seq_len x hid])
            enc_masks: input sequence masks (tensor of size [batch x seq_len])
            labels: dict storing
                type: action type labels (tensor of size [batch])
                arg: action argument labels (tensor of size [batch])
                mask: interactive mask selection labels (list of tensor [1] or None)
                interact: whether mask prediction is required (list of True/False)
                type_pos: list of positions to perform argument prediction (+1 for type positions)
                arg_pos: list of positions to perform argument prediction
                vis_len: list of length of visual inputs (i.e. object number +1)
        """
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device = self.device, non_blocking=True)


        lang_feats = batch['lang_input']
        vis_cls_col = batch['vision_cls']
        if self.args.drop_input != 0:
            lang_feats =drop_input(lang_feats, self.args.drop_input, self.training)
            vis_cls_col = drop_input(vis_cls_col, self.args.drop_input, self.training)
        his_feats = batch['action_history_input']

        vis_feat1 = self.action_emb(vis_cls_col)
        if not self.args.use_bert:
            vis_feat2 = batch['vision_feats']
            vis_feats = torch.cat([vis_feat1, vis_feat2], dim=2)
        else:
            vis_feat2 = self.vis_feat_to_emb(batch['vision_feats'])
            vis_feats = vis_feat1 + vis_feat2


        vis_his_feats = None
        if 'his_vis_cls' in batch:
            vis_his_feat1 = self.action_emb(batch['his_vis_cls'])   # batch x vis_his_len x obj_num x emb_dim
            vis_his_feat2 = batch['his_vis_feat']    # batch x vis_his_len x obj_num x 7
            vis_his_feats = torch.cat([vis_his_feat1, vis_his_feat2], dim=3).sum(dim=2) # batch x vis_his_len x (emb_dim+7)

        pos_feats = {'rotation': batch['rotation'], 'horizon': batch['horizon']} if 'rotation' in batch else None

        bs = len(lang_feats)
        feat2input = self.feats_to_inputs if not self.args.use_bert else self.feats_to_inputs_bert
        inputs = feat2input(lang_feats, vis_feats, his_feats, vis_his_feats, pos_feats)


        enc_masks = batch['seq_mask']  # batch x seq_len

        # cls_ = torch.ones((bs,1)).to(device=self.device, dtype=torch.long)
        # enc_masks = torch.cat([cls_, vis_cls_col, lang_feats, his_feats], dim=1)
        # enc_masks = (enc_masks != 0)
        # assert (enc_masks==batch['seq_mask']).sum() == enc_masks.shape[0]*enc_masks.shape[1]

        # print('vision:')
        # print(enc_masks[0][:32].data)
        # print(inputs[0][:10][:30].data)
        # print(vis_feat1[0][:10][:30].data)
        # print(batch['vision_feats'][0][:10].data)
        # print(vis_feat2[0][:10][:30].data)
        # print('language:')
        # print(lang_feats[0].data)
        # print(enc_masks[0][32:96].data)
        # print(inputs[0][32:66][:30].data)
        # print('action:')
        # print(enc_masks[0][96:].data)
        # print(inputs[0][96:117][:30].data)

        labels = {
            'type': batch['actype_label'],
            'arg': batch['arg_label'],
            'mask': batch['mask_label'],
            'interact': batch['interact'],
            'type_pos': batch['type_pos'],
            'arg_pos': batch['arg_pos'],
            }
        if 'visible_label' in batch:
            labels.update({
                'visible': batch['visible_label'],
                'reached': batch['reached_label'],
                'progress': batch['progress_label'],
                })

        return inputs, enc_masks, labels


    def feats_to_inputs(self, lang_feats, vis_feats, his_feats, vis_his_feats=None, pos_feats=None):
        """convert features (indexes, vision features) to transformer inputs (embeddings)

        Args:
            lang_feats: word ids of size (batch, lang_len)
            vis_feats: vision features of size (batch, vis_len, vis_feat_dim)
            his_feats: action ids of size (batch, his_len)

        Returns:
            inputs: transformer inputs of size (batch, seq_len, hid)
        """

        vis_in = self.vision_to_enc_input(vis_feats)   #batch x vis_len x hid
        lang_in = self.lang_to_enc_input(lang_feats)   # batch x lang_len x hid
        act_in = self.action_to_enc_input(his_feats)   # batch x action_len x hid

        if vis_his_feats is not None:
            vis_his_in = self.vis_his_to_enc_input(vis_his_feats)   # batch x (action_len/2-1) x hid
            act_in[:, 2::2, :] = vis_his_in

        if pos_feats is not None:
            ro = self.rotation_emb(pos_feats['rotation'])
            ho = self.horizon_emb(pos_feats['horizon'])
            cls_in = self.pos_ln(ro+ho).unsqueeze(1)
        else:
            bs, hid = len(lang_feats), self.args.hidden_dim
            cls_in = torch.zeros(bs, 1, hid).to(device=self.device)   #batch x 1 x hid

        token_emb = torch.cat([cls_in, vis_in, lang_in, act_in], dim=1)   # batch x seq_len x hid
        modal_emb = self.modality_emb(self.modal_indicator)
        pos_emb = self.positional_emb(self.position_indexes)
        inputs = self.dropout(token_emb + modal_emb + pos_emb)   # batch x seq_len x hid
        # inputs = token_emb + modal_emb + pos_emb   # batch x seq_len x hid

        return inputs


    def feats_to_inputs_bert(self, lang_feats, vis_feats, his_feats, vis_his_feats=None, pos_feats=None):
        """convert features (indexes, vision features) to transformer inputs (embeddings)

        Args:
            lang_feats: word ids of size (batch, lang_len)
            vis_feats: vision features of size (batch, vis_len, vis_feat_dim)
            his_feats: action ids of size (batch, his_len)

        Returns:
            inputs: transformer inputs of size (batch, seq_len, hid)
        """
        # vis_in = self.vision_to_emb_input(vis_feats)   #batch x vis_len x emb
        vis_in = vis_feats   #batch x vis_len x emb
        lang_in = self.word_emb(lang_feats)   # batch x lang_len x emb
        act_in = self.action_emb(his_feats)   # batch x action_len x emb

        if pos_feats is not None:
            ro = self.rotation_emb(pos_feats['rotation'])
            ho = self.horizon_emb(pos_feats['horizon'])
            cls_in = self.pos_ln(ro+ho).unsqueeze(1)
        else:
            bs, emb = len(lang_feats), self.args.emb_dim
            cls_in = torch.zeros(bs, 1, emb).to(device=self.device)   #batch x 1 x emb

        input_emb = torch.cat([cls_in, vis_in, lang_in, act_in], dim=1)   # batch x seq_len x emb
        # input_emb = self.input_emb(inputs_embeds=input_emb)   # batch x seq_len x hid

        return input_emb


    def outputs_to_logits(self, enc_output, attns_list, type_pos, arg_pos, task_type):

        bs = len(enc_output)
        enum = list(range(bs))

        if self.args.pred_head_pos == 'cls' and 'navi' in task_type:
            type_output = enc_output[:, 0, :]
            arg_output = enc_output[:, 0, :]
        else:
            type_output = enc_output[enum, type_pos]   # batch x hid
            arg_output = enc_output[enum, arg_pos]   # batch x hid


        weight_arg = self.action_emb(self.objs)   # num_obj x emb_dim
        arg_logits = self.fc3(arg_output).mm(weight_arg.t())
        # arg_logits = self.out_to_obj(arg_output)   # batch x num_obj
        mask_logits, navi_logits = [], {}
        if 'high' in task_type:
            weight_high = self.action_emb(self.high_actions)   # num_high x emb_dim
            type_logits = self.fc1(type_output).mm(weight_high.t())
            # type_logits = self.out_to_high(type_output)   # batch x num_high
        elif 'low' in task_type:
            weight_low = self.action_emb(self.low_actions)   # num_high x emb_dim
            type_logits = self.fc2(type_output).mm(weight_low.t())
            # type_logits = self.out_to_low(type_output)   # batch x num_low

            # if not self.args.use_bert:
            attns = attns_list[-1]  # last layer attns: batch x num_head x q_len x v_len
            attns = attns.sum(dim=1)  # sum/select of heads: b x q_len x v_len
            # attn of arg prediction token over each vision input
            if self.args.pred_head_pos == 'cls' and 'navi' in task_type:
                mask_logits = attns[:, 0, 1:(self.args.topk_objs+2)]   # batch x vis_len
            else:
                mask_logits = attns[enum, arg_pos, 1:(self.args.topk_objs+2)]   # batch x vis_len
            # else:
            #     mask_out = enc_output[enum, arg_pos]   #enc_output[:, 0, :]
            #     mask_logits = self.mask_prediction_head(mask_out)

            if self.args.auxiliary_loss_navi and 'navi' in task_type:
                cls_output = enc_output[:, 0, :]    # batch x hid
                navi_logits['visible'] = self.visible_monitor(cls_output)
                navi_logits['reached'] = self.reached_monitor(cls_output)
                navi_logits['progress'] = self.progress_monitor(cls_output)

        return type_logits, arg_logits, mask_logits, navi_logits



    def get_loss(self, type_logits, arg_logits, mask_logits, navi_logits, labels, task_type):
        mask_loss, navi_loss = self.zero, {}
        if self.args.focal_loss:
            bs = len(type_logits)
            enum = list(range(bs))

            type_probs = F.softmax(type_logits, dim=1)[enum, labels['type']]
            type_loss = F.cross_entropy(type_logits, labels['type'], reduction='none')
            type_loss = torch.mean(type_loss * (1-type_probs) ** self.args.focal_gamma)

            arg_probs = F.softmax(arg_logits, dim=1)[enum, labels['arg']]
            arg_loss = F.cross_entropy(arg_logits, labels['arg'], ignore_index=-1, reduction='none')
            arg_loss = torch.mean(arg_loss * (1-arg_probs) ** self.args.focal_gamma)

            if 'mani' in task_type:
                mask_probs = F.softmax(mask_logits, dim=1)[enum, labels['mask']]
                mask_loss = F.cross_entropy(mask_logits, labels['mask'], ignore_index=-1, reduction='none')
                mask_loss = torch.mean(mask_loss * (1-mask_probs) ** self.args.focal_gamma)

        else:
            type_loss = F.cross_entropy(type_logits, labels['type'])
            arg_loss = F.cross_entropy(arg_logits, labels['arg'], ignore_index=-1)
            # mask_loss = self.totensor(0.0)
            if 'mani' in task_type:
                mask_loss = F.cross_entropy(mask_logits, labels['mask'], ignore_index=-1)

        if self.args.auxiliary_loss_navi and 'navi' in task_type:
            l_v = F.binary_cross_entropy_with_logits(navi_logits['visible'].view(-1), labels['visible'], reduction='none')
            l_r = F.binary_cross_entropy_with_logits(navi_logits['reached'].view(-1), labels['reached'], reduction='none')
            l_p = 0.5 * (torch.sigmoid(navi_logits['progress']).view(-1) - labels['progress']).square()
            navi_loss['visible'] = l_v[labels['visible']!=-1].mean()
            navi_loss['reached'] = l_r[labels['reached']!=-1].mean()
            navi_loss['progress'] = l_p[labels['progress']!=-1].mean()

        return type_loss, arg_loss, mask_loss, navi_loss

    def pred(self, type_logits, arg_logits, mask_logits, navi_logits):
        type_preds = torch.argmax(type_logits, dim=1)
        arg_preds = torch.argmax(arg_logits, dim=1)
        mask_preds = torch.argmax(mask_logits, dim=1) if mask_logits != [] else []
        navi_preds = {}
        if navi_logits != {}:
            navi_preds['visible'] = (navi_logits['visible']>0.5).view(-1)
            navi_preds['reached'] = (navi_logits['reached']>0.5).view(-1)
            navi_preds['progress'] = torch.sigmoid(navi_logits['progress'].view(-1))
        return type_preds, arg_preds, mask_preds, navi_preds


    def step(self, observations, task_type, topk=1):
        # language
        lang_feats = self.lang_obs_to_feats(observations['lang'])

        # current vision
        vis_feat_dim = self.emb_dim+7 if not self.args.use_bert else self.emb_dim
        if observations['vis'] is None:
            vis_feats = torch.zeros((self.vis_len, vis_feat_dim), device=self.device)   # vis_len x (emb_dim+7)
            vis_cls_col = torch.zeros((self.vis_len, ), dtype=torch.long, device=self.device)   # vis_len
        else:
            vis_feats, vis_cls_col = self.vis_obs_to_feats(*observations['vis'])

        # history actions
        his_feats = self.action_history_to_feats(observations['act_his'])

        # history visions and posture
        if 'navi' not in task_type:
            vis_his_feats = pos_feats = None
        else:
            vis_his_feats = self.vis_his_obs_to_feats(observations['vis_his'])
            pos_feats = {k: self.totensor(v).unsqueeze(0) for k,v in observations['pos'].items()}

        feats_input = (lang_feats.unsqueeze(0), vis_feats.unsqueeze(0), his_feats.unsqueeze(0),
                              vis_his_feats, pos_feats)
        feat2input = self.feats_to_inputs if not self.args.use_bert else self.feats_to_inputs_bert
        inputs = feat2input(*feats_input)

        enc_masks = torch.cat([self.totensor([1]), vis_cls_col, lang_feats, his_feats], dim=0)
        enc_masks = (enc_masks != 0)
        # print('action_history', action_history)
        # print('enc_masks', enc_masks)

        if not self.args.use_bert:
            enc_output, attns_list = self.encoder(inputs, enc_masks)
            # enc_output: batch x seq_len x hid
        else:
            outputs = self.encoder(inputs_embeds=inputs, attention_mask=enc_masks.unsqueeze(0),
                                                  output_attentions=True)
            enc_output = outputs.last_hidden_state
            attns_list = outputs.attentions

        arg_pos = enc_masks.nonzero()[-1]
        type_pos = arg_pos - 1
        # print('type_pos', type_pos, 'arg_pos:', arg_pos)

        type_logits, arg_logits, mask_logits, _ = self.outputs_to_logits(enc_output, attns_list,
            type_pos, arg_pos, task_type)
            # type_pos.unsqueeze(0), arg_pos.unsqueeze(0), level)

        level = task_type.split('_')[0]
        type_probs, type_preds = self.topk_preds(type_logits, topk)
        type_preds = self.dec_vocabs['type_%s'%level].seq_decode(type_preds)
        arg_probs, arg_preds  = self.topk_preds(arg_logits, topk)
        arg_preds = self.dec_vocabs['arg'].seq_decode(arg_preds)
        if level == 'low':
            mask_probs, mask_preds = self.topk_preds(mask_logits, topk)
        else:
            mask_probs, mask_preds = [], []

        preds = {'type': type_preds, 'arg': arg_preds, 'mask': mask_preds}
        probs = {'type': type_probs, 'arg': arg_probs, 'mask': mask_probs}

        return preds, probs

    def topk_preds(self, logits, topk):
        probs = F.softmax(logits.squeeze(), dim=0)
        probs_sorted, idx_sorted = probs.topk(min(topk, len(probs)))
        return probs_sorted.tolist(), idx_sorted.tolist()


    def lang_obs_to_feats(self, lang_obs):
        lang_widx = torch.zeros((self.lang_len,), dtype=torch.long, device=self.device)   # lang_len
        if lang_obs is None:
            return lang_widx

        if self.args.use_bert:
            lang_obs = mmt_word_ids_to_bert_ids(lang_obs[1:-1], self.vocab, self.bert_tknz)
        actual_len = min(self.lang_len, len(lang_obs))
        lang_widx[:actual_len] = self.totensor(lang_obs[:actual_len])
        if not self.args.use_bert:
            lang_widx[lang_widx >= self.dataset.vocab.vocab_size] = 0
        # logging.info(lang_widx)
        return lang_widx   # lang_len


    def vis_obs_to_feats(self, boxes, classes, scores):
        obj_num = min(self.vis_len - 1, len(scores))

        vis_feat_col = torch.zeros((self.vis_len, 7), device=self.device)   # vis_len x 7
        vis_cls_col = torch.zeros((self.vis_len, ), dtype=torch.long, device=self.device)   # vis_len
        for obj_idx in range(obj_num):
            bbox = self.totensor(boxes[obj_idx]).to(torch.float) / self.image_size
            cls_idx = self.action_vocab.w2id(classes[obj_idx])
            vis_feat_col[obj_idx+1][:4] = bbox
            vis_feat_col[obj_idx+1][4] = bbox[2]-bbox[0]
            vis_feat_col[obj_idx+1][5] = bbox[3]-bbox[1]
            vis_feat_col[obj_idx+1][6] = float(scores[obj_idx])   # 1d
            vis_cls_col[obj_idx + 1] = cls_idx
        vis_cls_col[0] = 1
        cls_feat = self.action_emb(vis_cls_col)
        if not self.args.use_bert:
            vis_feat = torch.cat([cls_feat, vis_feat_col], dim=1)   #vis_len x (emb_dim+7)
        else:
            vis_feat = cls_feat + self.vis_feat_to_emb(vis_feat_col)
        # print('vis_feat:', vis_feat.shape)
        return vis_feat, vis_cls_col

    def vis_his_obs_to_feats(self, vis_his):
        if vis_his is None:
            return None
        # history visual input
        vis_his_len = self.vis_his_len = self.args.history_max_length - 1
        max_obj_num = self.max_obj_num = 10
        his_vis_feat = torch.zeros((vis_his_len, max_obj_num, 7), device=self.device)   # vis_his_len x max_obj_num x 7
        his_vis_cls = torch.zeros((vis_his_len, max_obj_num), dtype=torch.long, device=self.device)   # vis_his_len x max_obj_num
        for his_idx, dets in enumerate(vis_his[-vis_his_len:]):
            obj_num = min(max_obj_num, len(dets[0]))
            for obj_idx in range(obj_num):
                bbox = self.totensor(dets[0][obj_idx]).to(torch.float) / self.image_size
                cls_idx = self.action_vocab.w2id(dets[1][obj_idx])
                his_vis_feat[his_idx][obj_idx][:4] = bbox
                his_vis_feat[his_idx][obj_idx][4] = bbox[2]-bbox[0]
                his_vis_feat[his_idx][obj_idx][5] = bbox[3]-bbox[1]
                his_vis_feat[his_idx][obj_idx][6] = float(dets[2][obj_idx])
                his_vis_cls[his_idx][obj_idx] = cls_idx
        his_cls_feat = self.action_emb(his_vis_cls)   # vis_his_len x max_obj_num x emb_dim
        his_vis_feat_all = torch.cat([his_cls_feat, his_vis_feat], dim=2)   #vis_his_len x max_obj_num x (emb_dim+7)
        return his_vis_feat_all.sum(dim=1).unsqueeze(0)   #1 x vis_his_len x (emb_dim+7)

    def action_history_to_feats(self, action_history_seq):
        action_seq = torch.zeros((self.his_len,), dtype=torch.long, device=self.device)
        if action_history_seq is None:
            return action_seq
        elif isinstance(action_history_seq[0], str):
            action_history_seq = self.action_vocab.seq_encode(action_history_seq)
        actual_len = min(self.his_len, len(action_history_seq))
        action_seq[:actual_len] = self.totensor(action_history_seq[-actual_len:])
        # logging.info(action_seq)
        return action_seq   # his_len







