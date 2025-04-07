import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    def __init__(self, loss_weight, matcher):
        super().__init__()
        self.cost_relation = loss_weight["relation"]
        self.cost_head = loss_weight["head_entity"]
        self.matcher = matcher

    @torch.no_grad()
    def forward(self, outputs, targets):

        bsz, num_generated_triples = outputs["pred_rel_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, num_classes]
        gold_rel = torch.cat([v["relation"] for v in targets])
        # after masking the pad token
        pred_head_start = outputs["head_start_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
        pred_head_end = outputs["head_end_logits"].flatten(0, 1).softmax(-1)

        gold_head_start = torch.cat([v["head_start_index"] for v in targets])
        gold_head_end = torch.cat([v["head_end_index"] for v in targets])

        if self.matcher == "avg":
            cost = - self.cost_relation * pred_rel[:, gold_rel] - self.cost_head * 1/2 * (pred_head_start[:, gold_head_start] + pred_head_end[:, gold_head_end])
        elif self.matcher == "min":
            cost = torch.cat([pred_head_start[:, gold_head_start].unsqueeze(1), pred_rel[:, gold_rel].unsqueeze(1), pred_head_end[:, gold_head_end].unsqueeze(1)], dim=1)
            cost = - torch.min(cost, dim=1)[0]
        else:
            raise ValueError("Wrong matcher")
        cost = cost.view(bsz, num_generated_triples, -1).cpu()
        num_gold_triples = [len(v["relation"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(num_gold_triples, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]



# ## yyl

# import torch  
# from scipy.optimize import linear_sum_assignment  
# from torch import nn  
  
# # 假设的预测输出和真实标签  
# batch_size = 2  
# num_generated_triples = 4  
# num_classes = 3  
# seq_len = 10  
  
# # 模拟预测输出  
# outputs = {  
#     "pred_rel_logits": torch.randn(batch_size, num_generated_triples, num_classes),  
#     "head_start_logits": torch.randn(batch_size, num_generated_triples, seq_len),  
#     "head_end_logits": torch.randn(batch_size, num_generated_triples, seq_len),  
# }  
  
# # 模拟真实标签  
# targets = [  
#     {"relation": torch.tensor([0, 2]), "head_start_index": torch.tensor([1, 3]), "head_end_index": torch.tensor([2, 5])},  
#     {"relation": torch.tensor([1]), "head_start_index": torch.tensor([4]), "head_end_index": torch.tensor([6])},  
# ]  
  
# # 损失权重和匹配策略  
# loss_weight = {"relation": 1.0, "head_entity": 1.0}  
# matcher = "avg"  
  
# # 初始化匈牙利匹配器  
# hungarian_matcher = HungarianMatcher(loss_weight, matcher)  
  
# # 进行匹配  
# matches = hungarian_matcher(outputs, targets)  
  
# # 打印匹配结果  
# for i, (idx_pred, idx_true) in enumerate(matches):  
#     print(f"Batch {i}:")  
#     print(f"  Matched predictions: {idx_pred.tolist()}")  
#     print(f"  Matched targets: {idx_true.tolist()}")
