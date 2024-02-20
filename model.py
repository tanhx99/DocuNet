import torch, math
import torch.nn as nn
from opt_einsum import contract
from allennlp.modules.matrix_attention import DotProductMatrixAttention, CosineMatrixAttention, BilinearMatrixAttention
from losses import ATLoss, balanced_loss
from long_seq import process_long_input
from segmentation import AttentionUNet as SegNet



class DocREModel(nn.Module):
    def __init__(self, config, args, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.bert_model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = balanced_loss()

        self.featureDim = 256
        self.relation_dim = 256
        self.reduced_dim = config.hidden_size

        self.emb_size = emb_size
        self.block_size = block_size
        self.pos_label_num = num_labels

        self.unet_in_dim = args.unet_in_dim
        self.unet_out_dim = args.unet_in_dim
        self.min_height = args.max_height
        self.channel_type = args.channel_type

        self.linear = nn.Linear(config.hidden_size, 3)
        self.attention_weight = nn.Linear(self.reduced_dim, self.relation_dim)
        self.attention_net = nn.Parameter(torch.randn(config.num_labels, self.relation_dim))

        self.head_extractor = nn.Linear(1 * config.hidden_size+self.featureDim, config.hidden_size)
        self.tail_extractor = nn.Linear(1 * config.hidden_size+self.featureDim, config.hidden_size)

        # self.bilinear = nn.Bilinear(in1_features=self.featureDim, in2_features=self.featureDim, out_features=config.num_labels)
        self.bilinear = nn.Parameter(torch.randn(config.num_labels, config.hidden_size, config.hidden_size))
        self.bilinear_bais = nn.Parameter(torch.randn(config.num_labels))

        self.segmentation_net = SegNet(3, self.featureDim)
        nn.init.uniform_(self.bilinear,a=-math.sqrt(1/(2*self.reduced_dim)), b=math.sqrt(1/(2*self.reduced_dim)))
        nn.init.uniform_(self.bilinear_bais, a=-math.sqrt(1 / (2*self.reduced_dim)), b=math.sqrt(1 / (2*self.reduced_dim)))
        nn.init.xavier_normal_(self.attention_net)



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

    def get_entity_embs_and_attn(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        bs, h, _, max_len = attention.size()
        # ne = max([len(x) for x in entity_pos])  # 本次bs中的最大实体数
        ne = self.min_height    # 数据集中最大实体数量
        nm = 0  # 本次bs中实体的最大提及数，在下面的程序中得到
        for b in range(len(entity_pos)):
            mention_num = [len(x) for x in entity_pos[b]]
            nm = max(nm, max(mention_num))

        entity_es = []  # size(bs, ne, dim)
        entity_as = []  # size(bs, ne, head, dim)
        entity_mention_num = []

        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for entity_num, e in enumerate(entity_pos[i]):
                e_emb, e_att = [], []   # 对单个实体来说
                for start, end, sent_id in e:
                    mention_emb = None
                    mention_attn = None
                    if start + offset < max_len:
                        # In case the entity mention is truncated due to limited max seq length.
                        mention_emb = sequence_output[i, start + offset]
                        mention_attn = attention[i, :, start + offset]
                    else:
                        mention_emb = torch.zeros(self.config.hidden_size).to(sequence_output)  # 提及位置超过了最大长度
                        mention_attn = torch.zeros(h, max_len).to(attention)
                    e_emb.append(mention_emb)
                    e_att.append(mention_attn)
                
                e_att = torch.stack(e_att, dim=0).mean(0)
                entity_atts.append(e_att)


                for _ in range(nm-len(e_emb)):
                    masked_mention_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                    e_emb.append(masked_mention_emb)
                e_emb = torch.stack(e_emb, dim=0)

                entity_embs.append(e_emb)   # e_emb是一个实体的embedding
                
            entity_mention_num.append([len(x) for x in entity_pos[i]])

            for _ in range(ne-entity_num-1):
                masked_mention_attn = torch.zeros(h, max_len).to(attention)
                entity_atts.append(masked_mention_attn)
                entity_embs.append(torch.zeros((nm, self.config.hidden_size)).to(sequence_output))

            entity_embs = torch.stack(entity_embs, dim=0)
            entity_atts = torch.stack(entity_atts, dim=0)
            entity_es.append(entity_embs)
            entity_as.append(entity_atts)
        
        entity_es = torch.stack(entity_es, dim=0)
        ent_men_mask = torch.zeros((bs, ne, nm), device=sequence_output.device)
        for i in range(bs):
            for j in range(len(entity_mention_num[i])):
                ent_men_mask[i, j, :entity_mention_num[i][j]] = 1.

        men_mask = (1.0 - ent_men_mask.unsqueeze(-1)) * -1000000.0
        # get relation-specific attention for each mention
        # [bs, max_ent_cnt*max_men_cnt,reduced_dim] * [reduced_dim,num_labels]
        men_attention = torch.matmul(nn.Tanh()(self.attention_weight(entity_es.view(bs, -1, self.reduced_dim))), self.attention_net.transpose(0,1).contiguous()) # shape:[bs, max_ent_cnt*max_men_cnt, num_labels]
        # add mask before softmax
        men_attention = men_attention.view(bs, ne, nm, -1) + men_mask # shape:[bs, max_ent_cnt, max_men_cnt, num_labels]
        men_attention = nn.Softmax(dim=-2)(men_attention).permute(0,1,3,2).contiguous() # shape:[bs, max_ent_cnt, num_labels, max_men_cnt]

        # get relation-specific rep for each entity
        entity_es = torch.matmul(men_attention, entity_es) # shape:[bs, max_ent_cnt, num_labels, reduced_dim]

        return entity_es, entity_as


    def get_hts(self, entity_embs, rel_enco, hts):
        htss, hss, tss = [], [], []
        for i in range(len(hts)):
            ht_index = hts[i]
            for (h_index, t_index) in ht_index:
                htss.append(rel_enco[i,h_index,t_index])
                hss.append(entity_embs[i][h_index])
                tss.append(entity_embs[i][t_index])
        htss = torch.stack(htss, dim=0)
        hss = torch.stack(hss, dim=0)
        tss = torch.stack(tss, dim=0)

        return hss, tss, htss

    def get_channel_map(self, sequence_output, entity_as):
        bs,_,d = sequence_output.size()
        # ne = max([len(x) for x in entity_as])  # 本次bs中的最大实体数
        ne = self.min_height

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

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)

        bs, sequen_len, d = sequence_output.shape

        # get hs, ts and entity_embs >> entity_rs
        entity_embs, entity_as = self.get_entity_embs_and_attn(sequence_output, attention, entity_pos, hts)

        # 获得通道map的两种不同方法
        if self.channel_type == 'context-based':
            feature_map = self.get_channel_map(sequence_output, entity_as)  # [b, min_height, min_height, d]
            attn_input = self.linear(feature_map).permute(0, 3, 1, 2).contiguous()  # [b, 3, min_height, min_height] 3个通道

        elif self.channel_type == 'similarity-based':
            ent_encode = sequence_output.new_zeros(bs, self.min_height, d)
            for _b in range(bs):
                entity_emb = entity_embs[_b]
                entity_num = entity_emb.size(0)
                ent_encode[_b, :entity_num, :] = entity_emb
            # similar0 = ElementWiseMatrixAttention()(ent_encode, ent_encode).unsqueeze(-1)
            similar1 = DotProductMatrixAttention()(ent_encode, ent_encode).unsqueeze(-1)
            similar2 = CosineMatrixAttention()(ent_encode, ent_encode).unsqueeze(-1)
            similar3 = BilinearMatrixAttention(self.emb_size, self.emb_size).to(ent_encode.device)(ent_encode, ent_encode).unsqueeze(-1)
            attn_input = torch.cat([similar1,similar2,similar3],dim=-1).permute(0, 3, 1, 2).contiguous()
        else:
            raise Exception("channel_type must be specify correctly")

        attn_map = self.segmentation_net(attn_input).permute(0,2,3,1).contiguous()
        hs, ts, h_t = self.get_hts(entity_embs, attn_map, hts)

        h_t = h_t.unsqueeze(1).repeat(1, hs.shape[1], 1)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, h_t], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, h_t], dim=-1)))

        # logits = self.bilinear(hs, ts)
        logits = torch.einsum("nkd,kdp,nkp->nk", [hs, self.bilinear, ts]) + self.bilinear_bais

        output = dict()
        pred = (self.loss_fnt.get_label(logits, num_labels=self.pos_label_num))
        output["pred"] = pred
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = loss.to(sequence_output)
        return output



