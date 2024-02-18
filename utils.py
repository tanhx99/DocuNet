import torch
import random
import numpy as np
from allennlp.modules.matrix_attention import DotProductMatrixAttention, CosineMatrixAttention, BilinearMatrixAttention


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    entity_pos = [f["entity_pos"] for f in batch]

    labels = [f["labels"] for f in batch]
    hts = [f["hts"] for f in batch]
    output = (input_ids, input_mask, labels, entity_pos, hts)
    return output


def getEntityEmbeddings(sequence_output, entityPos):
    assert(len(sequence_output) == len(entityPos))
    maxSeqLenOfABatch = sequence_output.shape[1]
    batchSize = len(entityPos)
    batchEntityEmbeddings = []
    for i in range(batchSize):
        entityEmbeddings = []
        for entityID, mentionPos in enumerate(entityPos[i]):
            entityEmbedding = None
            if len(mentionPos) == 1: # 表示这个实体只有一个提及
                start, end = mentionPos[0]
                if start + 1 < maxSeqLenOfABatch:
                    entityEmbedding = sequence_output[i, start + 1]  # entityEmbedding里面有一个tensor的元素
                if len(entityEmbedding) == 0:
                    entityEmbedding = torch.zeros(768)
            #到这里为止，entityEmbedding是一个包含了一个tensor的列表
            else:  # 这个实体有多个提及
                mentionEmbedding = []
                for start, end in mentionPos:
                    if start + 1 < maxSeqLenOfABatch:
                        mentionEmbedding.append(sequence_output[i, start + 1])
                if len(mentionEmbedding) == 0:
                    mentionEmbedding.append(torch.zeros(768))
                else:
                    mentionEmbedding = torch.stack(mentionEmbedding, dim=0)
                    entityEmbedding = torch.logsumexp(mentionEmbedding, dim=0)
                    
            entityEmbeddings.append(entityEmbedding)
        entityEmbeddings = torch.stack(entityEmbeddings)
        batchEntityEmbeddings.append(entityEmbeddings) # batchEntityEmbeddings每个元素的形状应该是：[entityNum, 768]
    return batchEntityEmbeddings

def getFeatureMap(sequence_output, batchEntityEmbeddings):
    # batchEntityEmbeddings = getEntityEmbeddings(sequence_output, entityPos)
    
    b, _, d = sequence_output.shape
    ent_encode = sequence_output.new_zeros(b, 42, d)
    for _b in range(b):
        entity_emb = batchEntityEmbeddings[_b]     #实体的embedding
        entity_num = entity_emb.size(0)
        ent_encode[_b, :entity_num, :] = entity_emb    # entity_emb : (entity_num, 768)
    # similar0 = ElementWiseMatrixAttention()(ent_encode, ent_encode).unsqueeze(-1)
    similar1 = DotProductMatrixAttention()(ent_encode, ent_encode).unsqueeze(-1) # (4,42,42,1)
    similar2 = CosineMatrixAttention()(ent_encode, ent_encode).unsqueeze(-1)
    similar3 = BilinearMatrixAttention(768, 768).to(ent_encode.device)(ent_encode, ent_encode).unsqueeze(-1)
    attn_input = torch.cat([similar1,similar2,similar3],dim=-1).permute(0, 3, 1, 2).contiguous()
    return attn_input

#应该返回的形状:(batch_size, 97, 42, 42)
def getLabelMap(labels, hts):
    assert(len(labels) == len(hts))
    labelMaps = []
    for i in range(len(labels)):
        assert(len(labels[i]) == len(hts[i]))
        labelMap = np.zeros(shape=(42, 42, 97), dtype=np.int32)
        labelMap[..., 0] = 1
        labelMap = torch.from_numpy(labelMap)
        # labelMap = torch.zeros(size=(42, 42, 97), dtype=torch.int32)
        for j in range(len(labels[i])):
            l = torch.tensor(labels[i][j])
            h = hts[i][j][0]
            t = hts[i][j][1]
            labelMap[h, t, :] = l
        labelMap = labelMap.permute(2, 0, 1).contiguous()
        labelMaps.append(labelMap)
    labelMaps = torch.stack(labelMaps)
    labelMaps = labelMaps.to(dtype=torch.float)
    return labelMaps

def getGraphEdges(size):
    edges = [[],[]]
    for i in range(size[0]):
        for j in range(size[1]):
            if i - 1 >= 0: # 不是最上边儿的元素
                edges[0].append(i * size[0] + j)
                edges[1].append((i - 1) * size[0])
            if i + 1 < size[0]: # 不是最下边儿的元素
                edges[0].append(i * size[0] + j)
                edges[1].append((i + 1) * size[0] + j)
            if j - 1 >= 0: # 不是最左边儿的元素
                edges[0].append(i * size[0] + j)
                edges[1].append(i * size[0] + j - 1)
            if j + 1 < size[1]: #不是最右边儿的元素
                edges[0].append(i * size[0] + j)
                edges[1].append(i * size[0] + j + 1)
    return torch.tensor(edges, dtype=torch.long)
