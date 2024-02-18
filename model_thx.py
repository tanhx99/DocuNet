import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from torch_geometric.nn import GCNConv
from utils import getGraphEdges
from losses import ATLoss, balanced_loss
from long_seq import process_long_input


class GCNet(nn.Module):
    def __init__(self, inChannels : int, outChannels : int) -> None:
        super().__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.gcn1 = GCNConv(in_channels=self.inChannels, out_channels=128)
        self.gcn2 = GCNConv(in_channels=128, out_channels=self.outChannels)
        # self.weights = nn.Parameter()

    def forward(self, inFeatures):  # inFeatures : (batchSize, inChannels, 42, 42)
        batchFeatures = []
        for item in inFeatures: # item : (inChannels, 42, 42)
            permutedItem = item.permute(1, 2, 0).contiguous() # permutedItem : (42, 42, inChannels)
            flattenItem = torch.flatten(permutedItem, end_dim=1)
            edges = getGraphEdges(size=(permutedItem.shape[0], permutedItem.shape[1]))
            edges = edges.to(item.device)
            out = self.gcn1(flattenItem, edges)
            out = self.gcn2(out, edges)
            out = torch.reshape(input=out, shape=(self.outChannels, 35, 35))
            batchFeatures.append(out)
        batchFeatures = torch.stack(batchFeatures)
        return batchFeatures




class DocREModel(nn.Module):
    def __init__(self, config, args, model, num_labels=-1):
        super().__init__()
        self.config = config
        self.bert_model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = balanced_loss()

        self.featureDim = 512
        self.num_labels = num_labels

        self.min_height = args.max_height
        self.channel_type = args.channel_type

        self.linear = nn.Linear(config.hidden_size, 3)

        self.head_extractor = nn.Linear(1 * config.hidden_size, self.featureDim // 2)
        self.tail_extractor = nn.Linear(1 * config.hidden_size, self.featureDim // 2)

        self.bilinear = nn.Bilinear(in1_features=self.featureDim // 2, in2_features=self.featureDim // 2, out_features=config.num_labels)



    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.bert_model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        bs, h, _, max_len = attention.size()
        # ne = max([len(x) for x in entity_pos])  # 本次bs中的最大实体数

        max_node_num = 0
        for b in range(len(entity_pos)):
            node_num = 1 + len(entity_pos[b]) + sum([len(x) for x in entity_pos[b]])
            max_node_num = max(node_num, max_node_num)

        doc_edge = torch.zeros((bs, max_node_num, max_node_num), device=sequence_output.device)
        intra_ent_edge = torch.zeros((bs, max_node_num, max_node_num), device=sequence_output.device)
        inter_ent_edge = torch.zeros((bs, max_node_num, max_node_num), device=sequence_output.device)

        hss, tss, rss = [], [], []
        entity_es = []
        entity_as = []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for entity_num, e in enumerate(entity_pos[i]):
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end, sent_id in e:
                        if start + offset < max_len:
                            # In case the entity mention is truncated due to limited max seq length.
                            mention_emb = sequence_output[i, start + offset]
                            e_emb.append(mention_emb)
                            e_att.append(attention[i, :, start + offset])
                        else:
                            mention_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, max_len).to(attention)
                else:
                    start, end, sent_id = e[0]
                    if start + offset < max_len:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, max_len).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
            for _ in range(self.min_height-entity_num-1):
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]


            entity_es.append(entity_embs)
            entity_as.append(entity_atts)
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            hss.append(hs)
            tss.append(ts)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        return hss, tss, entity_es, entity_as

    def get_ht(self, rel_enco, hts):
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for (h_index, t_index) in ht_index:
                htss.append(rel_enco[i,h_index,t_index])
        htss = torch.stack(htss,dim=0)
        return htss

    def get_channel_map(self, sequence_output, entity_as):
        # sequence_output = sequence_output.to('cpu')
        # attention = attention.to('cpu')
        bs,_,d = sequence_output.size()
        # ne = max([len(x) for x in entity_as])  # 本次bs中的最大实体数
        ne = self.min_height    # 这里应该是采用了固定值

        index_pair = []
        for i in range(ne):
            tmp = torch.cat((torch.ones((ne, 1), dtype=int) * i, torch.arange(0, ne).unsqueeze(1)), dim=-1)
            index_pair.append(tmp)
        index_pair = torch.stack(index_pair, dim=0).reshape(-1, 2).to(sequence_output.device)
        map_rss = []
        for b in range(bs):
            entity_atts = entity_as[b]
            h_att = torch.index_select(entity_atts, 0, index_pair[:, 0])
            t_att = torch.index_select(entity_atts, 0, index_pair[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[b], ht_att)
            map_rss.append(rs)
        map_rss = torch.cat(map_rss, dim=0).reshape(bs, ne, ne, d)
        return map_rss

    def forward(self, input_ids=None, attention_mask=None, labels=None, entity_pos=None, hts=None):

        sequence_output, attention = self.encode(input_ids, attention_mask)


        # get hs, ts and entity_embs >> entity_rs
        hs, ts, entity_embs, entity_as = self.get_hrt(sequence_output, attention, entity_pos, hts)

        feature_map = self.get_channel_map(sequence_output, entity_as)  # [b, min_height, min_height, d]
        attn_input = self.linear(feature_map).permute(0, 3, 1, 2).contiguous()  # [b, 3, min_height, min_height] 3个通道


        attn_map = self.segmentation_net(attn_input).permute(0,2,3,1).contiguous()
        h_t = self.get_ht(attn_map, hts)

        hs = torch.tanh(self.head_extractor(hs) + h_t)
        ts = torch.tanh(self.tail_extractor(ts) + h_t)

        logits = self.bilinear(hs, ts)

        output = dict()
        pred = (self.loss_fnt.get_label(logits, num_labels=self.num_labels))
        output["pred"] = pred
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = loss.to(sequence_output)
        return output



