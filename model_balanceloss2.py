import torch
import torch.nn as nn
import math
from opt_einsum import contract
from long_seq import process_long_input
from losses import balanced_loss as ATLoss
import torch.nn.functional as F
from attn_unet import AttentionUNet


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000.0 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionDocREModel(nn.Module):
    def __init__(self, config, args, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.bert_model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(1 * config.hidden_size + args.unet_out_dim, emb_size)
        self.tail_extractor = nn.Linear(1 * config.hidden_size + args.unet_out_dim, emb_size)
        # self.head_extractor = nn.Linear(1 * config.hidden_size , emb_size)
        # self.tail_extractor = nn.Linear(1 * config.hidden_size , emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.bertdrop = nn.Dropout(0.6)
        self.unet_in_dim = args.unet_in_dim
        self.unet_out_dim = args.unet_in_dim
        self.linear = nn.Linear(config.hidden_size, args.unet_in_dim)
        self.min_height = args.max_height
        self.channel_type = args.channel_type
        self.segmentation_net = AttentionUNet(input_channels=args.unet_in_dim,
                                              class_number=args.unet_out_dim,
                                              down_channel=args.down_dim)
        # diffusion-related
        self.num_timesteps = args.num_timesteps
        self.sampling_timesteps = args.sampling_timesteps
        self.ddim_sampling_eta = 1.
        betas = linear_beta_schedule(self.num_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # self.F = nn.Sequential(
        #     nn.Linear(1, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size, 3)
        # )

        self.F = nn.Conv2d(1, 3, kernel_size=3, padding=1)    # 论文里给的好像是这样的



    def encode(self, input_ids, attention_mask,entity_pos):
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
        bs, h, _, c = attention.size()
        # ne = max([len(x) for x in entity_pos])  # 本次bs中的最大实体数

        hss, tss, rss = [], [], []
        entity_es = []
        entity_as = []
        x_0s = []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for entity_num, e in enumerate(entity_pos[i]):
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end, sent_id in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end, sent_id = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
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

            x0 = torch.zeros((self.min_height, self.min_height)).to(sequence_output.device)
            
            for (head, tail) in hts[i]:
                x0[head, tail] = 1
            x0[(entity_num+1):] = x0[entity_num]

            hss.append(hs)
            tss.append(ts)
            x_0s.append(x0)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        x_0s = torch.stack(x_0s, dim=0)
        return hss, tss, entity_es, entity_as, x_0s
    

    def prepare_targets(self, x_0s):
        diffused_xts = []
        noises = []
        ts = []
        for x0 in x_0s:
            t = torch.randint(0, self.num_timesteps, (1,)).long().to(x0.device)
            noise = torch.randn((self.min_height, self.min_height)).to(x0.device)
            x0 = x0 * 2. - 1.   # [0, 1] --> [-1, 1]
            xt = self.q_sample(x_start=x0, t=t, noise=noise)
            xt = torch.clamp(xt, min=-1, max=1)
            xt = (xt + 1) / 2. # [-1, 1] --> [0, 1]
            diff_xt = torch.clamp(xt, min=0, max=1)

            diffused_xts.append(diff_xt)
            noises.append(noise)
            ts.append(t)
        return torch.stack(diffused_xts), torch.stack(noises), torch.stack(ts)


    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def get_mask(self, ents, bs, ne, run_device):
        ent_mask = torch.zeros(bs, ne, device=run_device)
        rel_mask = torch.zeros(bs, ne, ne, device=run_device)
        for _b in range(bs):
            ent_mask[_b, :len(ents[_b])] = 1
            rel_mask[_b, :len(ents[_b]), :len(ents[_b])] = 1
        return ent_mask, rel_mask

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

    def model_predictions(self, sequence_output, hts, xt, hs, ts, entity_as, entity_nums, timestep):
        xt = torch.clamp(xt, min=-1, max=1)
        xt = (xt + 1) / 2
        xt = torch.clamp(xt, min=0, max=1)

        # xts = self.F(xt.unsqueeze(-1))
        xts = self.F(xt.unsqueeze(1)).permute(0, 2, 3, 1).contiguous()
        feature_map = self.get_channel_map(sequence_output, entity_as)  # [b, min_height, min_height, d]
        attn_input = self.linear(feature_map)  # [b, min_height, min_height, 3] 3个通道
        attn_input = attn_input + xts
        attn_input = attn_input.permute(0, 3, 1, 2).contiguous()
        attn_map = self.segmentation_net(attn_input, timestep)
        h_t = self.get_ht(attn_map, hts)
        hs = torch.tanh(self.head_extractor(torch.cat([hs, h_t], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, h_t], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        pred = (self.loss_fnt.get_label(logits, num_labels=self.num_labels))

        bs = len(hts)
        x0 = torch.zeros((bs, self.min_height, self.min_height)).to(sequence_output.device)

        last_i = 0
        for i in range(bs):
            length = len(hts[i])
            p = pred[last_i: last_i+length, 1]
            last_i += length
            for j in range(length):
                head, tail = hts[i][j]
                x0[i][head][tail] = p[j]
            entity_num = entity_nums[i]
            x0[i, entity_num:] = x0[i][entity_num-1]

        x_start = x0 * 2. - 1.
        x_start = torch.clamp(x_start, min=-1, max=1)
        pred_noise = self.predict_noise_from_start(xt, timestep, x_start) # epsilon sharp tao

        return pred_noise, x_start, logits, pred

    @torch.no_grad()
    def ddim_sample(self, sequence_output, hts, hs, ts, entity_as, entity_nums):
        batch = len(entity_as)
        device = hs.device
        shape = (batch, self.min_height, self.min_height)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        xt = torch.randn(shape).to(device)

        x_start = None
        for time, time_next in time_pairs:
            timestep = torch.full((batch,), time, device=device, dtype=torch.long)

            pred_noise, x_start, logits, pred = self.model_predictions(sequence_output, hts, xt, hs, ts, entity_as, entity_nums, timestep)

            if time_next < 0:
                xt = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(xt)

            xt = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise


        output = {"pred": pred}
        return output




    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask, entity_pos)

        bs, sequen_len, d = sequence_output.shape
        hs, ts, entity_embs, entity_as, x_0s = self.get_hrt(sequence_output, attention, entity_pos, hts)
        
        if not self.training:
            results = self.ddim_sample(sequence_output, hts, hs, ts, entity_as, [len(x) for x in entity_pos])
            return results
        

        xts, noises, times = self.prepare_targets(x_0s) # xts.size(bs, min_height, min_height)
        times = times.squeeze(-1) # (bs, )
        
        # xts = self.F(xts.unsqueeze(-1))
        xts = self.F(xts.unsqueeze(1)).permute(0, 2, 3, 1).contiguous()

        feature_map = self.get_channel_map(sequence_output, entity_as)  # [b, min_height, min_height, d]
        attn_input = self.linear(feature_map)  # [b, min_height, min_height, 3] 3个通道
        attn_input = attn_input + xts
        attn_input = attn_input.permute(0, 3, 1, 2).contiguous()

        attn_map = self.segmentation_net(attn_input, times)
        h_t = self.get_ht(attn_map, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, h_t], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, h_t], dim=1)))


        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)


        output = dict()
        pred = (self.loss_fnt.get_label(logits, num_labels=self.num_labels))
        output["pred"] = pred
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = loss.to(sequence_output)
        return output




