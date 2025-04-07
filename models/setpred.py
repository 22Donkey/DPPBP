import torch.nn as nn
import torch
from models.set_decoder import SetDecoder
from models.set_criterion import SetCriterion
from models.seq_encoder import SeqEncoder
from utils.functions import generate_triple
import copy


class SetPred(nn.Module):

    def __init__(self, args, num_classes):
        super(SetPred, self).__init__()
        self.args = args
        self.encoder = SeqEncoder(args)
        config = self.encoder.config
        self.num_classes = num_classes  # 关系抽取任务中关系类别的数量

        # config用于定义解码器结构 生成的三元组数量 解码器的层数 关系类别数量 是否返回中间解码层的输出
        self.decoder = SetDecoder(config, args.num_generated_triples, args.num_decoder_layers, num_classes, return_intermediate=False)
        # 不同损失项的权重 非关系类别的系数 计算那些损失（实体，关系) 匹配器将三元组与实际标签对齐
        self.criterion = SetCriterion(num_classes,  loss_weight=self.get_loss_weight(args), na_coef=args.na_rel_coef, losses=["entity", "relation"], matcher=args.matcher)

    def forward(self, input_ids, attention_mask, targets=None):
        # 输入词的索引和掩码，通过编码器对输入进行编码，返回最后一层的隐藏状态和序列的整体表示
        last_hidden_state, pooler_output = self.encoder(input_ids, attention_mask) 

        # 输入隐藏状态和注意力掩码 输出关系类别的预测，实体头部(subject)的起止位置预测
        class_logits, head_start_logits, head_end_logits = self.decoder(encoder_hidden_states=last_hidden_state, encoder_attention_mask=attention_mask)
        # head_start_logits, head_end_logits = span_logits.split(1, dim=-1)
        head_start_logits = head_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        head_end_logits = head_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)# [bsz, num_generated_triples, seq_len]
        outputs = {'pred_rel_logits': class_logits, 'head_start_logits': head_start_logits, 'head_end_logits': head_end_logits}
        if targets is not None:
            loss = self.criterion(outputs, targets)
            return loss, outputs
        else:
            return outputs

    def gen_triples(self, input_ids, attention_mask, info):
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            # print(outputs)
            pred_triple = generate_triple(outputs, info, self.args, self.num_classes)
            # print(pred_triple)
        return pred_triple

    # 将一个批次的数据组织成input_ids、attention_mask、targets格式，主要作用将不定长的输入序列填充为定长的张量，并生成相应的注意力掩码
    def batchify(self, batch_list):
        # print("Batch List:", batch_list)  # 打印batch_list内容
        # input('stop')
        batch_size  = len(batch_list)
        sent_idx = [ele[0] for ele in batch_list]
        sent_ids = [ele[1] for ele in batch_list]
        targets = [ele[2] for ele in batch_list]
        sent_lens = list(map(len, sent_ids))
        max_sent_len = max(sent_lens)

        input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
        
        for idx, (seq, seqlen) in enumerate(zip(sent_ids, sent_lens)):
            input_ids[idx, :seqlen] = torch.LongTensor(seq)
            attention_mask[idx, :seqlen] = torch.FloatTensor([1] * seqlen)
            
        if self.args.use_gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            targets = [{k: torch.tensor(v, dtype=torch.long, requires_grad=False).cuda() for k, v in t.items()} for t in targets]
        else:
            targets = [{k: torch.tensor(v, dtype=torch.long, requires_grad=False) for k, v in t.items()} for t in targets]
        info = {"seq_len": sent_lens, "sent_idx": sent_idx}
        return input_ids, attention_mask, targets, info



    @staticmethod
    def get_loss_weight(args):
        return {"relation": args.rel_loss_weight, "head_entity": args.head_ent_loss_weight}


