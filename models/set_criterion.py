import torch.nn.functional as F
import torch.nn as nn
import torch, math
from models.matcher import HungarianMatcher


class SetCriterion(nn.Module):
    
    def __init__(self, num_classes, loss_weight, na_coef, losses, matcher):       
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.matcher = HungarianMatcher(loss_weight, matcher)
        self.losses = losses
        rel_weight = torch.ones(self.num_classes + 1)
        rel_weight[-1] = na_coef
        self.register_buffer('rel_weight', rel_weight)

    def forward(self, outputs, targets):
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == "entity" and self.empty_targets(targets):
                pass
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices))
        losses = sum(losses[k] * self.loss_weight[k] for k in losses.keys() if k in self.loss_weight)
        return losses

    def relation_loss(self, outputs, targets, indices):
        src_logits = outputs['pred_rel_logits'] # [bsz, num_generated_triples, num_rel+1]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=self.rel_weight)
        losses = {'relation': loss}
        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        pred_rel_logits = outputs['pred_rel_logits']
        device = pred_rel_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_rel_logits.argmax(-1) != pred_rel_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices,  **kwargs):
        loss_map = {
            'relation': self.relation_loss,
            'cardinality': self.loss_cardinality,
            'entity': self.entity_loss
        }
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def entity_loss(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        selected_pred_head_start = outputs["head_start_logits"][idx]
        selected_pred_head_end = outputs["head_end_logits"][idx]

        target_head_start = torch.cat([t["head_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_head_end = torch.cat([t["head_end_index"][i] for t, (_, i) in zip(targets, indices)])


        head_start_loss = F.cross_entropy(selected_pred_head_start, target_head_start)
        head_end_loss = F.cross_entropy(selected_pred_head_end, target_head_end)
        losses = {'head_entity': 1/2*(head_start_loss + head_end_loss)}
        return losses

    @staticmethod
    def empty_targets(targets):
        flag = True
        for target in targets:
            if len(target["relation"]) != 0:
                flag = False
                break
        return flag


# ## yyl
# import torch

# # 假设的模型输出和目标数据
# def main():
#     # 模拟的模型输出
#     outputs = {
#         "pred_rel_logits": torch.tensor([
#             [[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]],  # 第一个样本的两个预测
#             [[0.3, 0.4, 0.3], [0.2, 0.3, 0.5]]   # 第二个样本的两个预测
#         ]),  # shape: [batch_size, num_generated_triples, num_rel + 1]
#         "head_start_logits": torch.tensor([
#             [[0.2, 0.8], [0.3, 0.7]],  # 第一个样本的两个头实体开始位置预测
#             [[0.6, 0.4], [0.1, 0.9]]   # 第二个样本的两个头实体开始位置预测
#         ]),  # shape: [batch_size, num_generated_triples, seq_len]
#         "head_end_logits": torch.tensor([
#             [[0.5, 0.5], [0.6, 0.4]],  # 第一个样本的两个头实体结束位置预测
#             [[0.4, 0.6], [0.3, 0.7]]   # 第二个样本的两个头实体结束位置预测
#         ]),  # shape: [batch_size, num_generated_triples, seq_len]
#     }

#     # 模拟的目标数据
#     targets = [
#         {
#             "relation": torch.tensor([2, 0]),  # 第一个样本的关系
#             "head_start_index": torch.tensor([0, 1]),  # 第一个样本头实体开始位置
#             "head_end_index": torch.tensor([1, 0])    # 第一个样本头实体结束位置
#         },
#         {
#             "relation": torch.tensor([1, 2]),  # 第二个样本的关系
#             "head_start_index": torch.tensor([0, 1]),
#             "head_end_index": torch.tensor([1, 0])
#         }
#     ]

#     # 设置损失权重和参数
#     num_classes = 2  # 类别数量（不包括 NA 类别）
#     loss_weight = {
#         "relation": 1.0,
#         "head_entity": 1.0,
#         "cardinality": 1.0
#     }
#     na_coef = 1.0
#     # 创建权重张量
#     rel_weight = torch.ones(num_classes + 1)  # num_classes + 1 以包含 NA 类别
#     rel_weight[-1] = na_coef  # 设置 NA 类别的权重

#     # 创建 SetCriterion 实例
#     criterion = SetCriterion(num_classes, loss_weight, na_coef, ["relation", "entity"], "avg")

#     # 将 rel_weight 注册到 criterion 中
#     criterion.register_buffer('rel_weight', rel_weight)

#     # 计算损失
#     loss = criterion(outputs, targets)

#     print(f"Calculated loss: {loss.item()}")

# if __name__ == "__main__":
#     main()




















# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=1.5, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets, weight=None):
#         logpt = F.log_softmax(inputs, dim=-1)
#         pt = torch.exp(logpt)
#         logpt = logpt.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
#         pt = pt.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
#         loss = -self.alpha * (1 - pt) ** self.gamma * logpt
#         if weight is not None:
#             loss = loss * weight[targets]
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         return loss


# class SetCriterion(nn.Module):
#     def __init__(self, num_classes, loss_weight, na_coef, losses, matcher):
#         super().__init__()
#         self.num_classes = num_classes
#         self.loss_weight = loss_weight
#         self.matcher = HungarianMatcher(loss_weight, matcher)
#         self.losses = losses
#         rel_weight = torch.ones(self.num_classes + 1)
#         rel_weight[-1] = na_coef
#         self.register_buffer('rel_weight', rel_weight)
#         # Use Focal Loss
#         self.focal_loss = FocalLoss(alpha=0.25, gamma=1.5)

#     def forward(self, outputs, targets):
#         indices = self.matcher(outputs, targets)
#         losses = {}
#         for loss in self.losses:
#             if loss == "entity" and self.empty_targets(targets):
#                 pass
#             else:
#                 losses.update(self.get_loss(loss, outputs, targets, indices))
#         losses = sum(losses[k] * self.loss_weight[k] for k in losses.keys() if k in self.loss_weight)
#         return losses

#     def relation_loss(self, outputs, targets, indices):
#         src_logits = outputs['pred_rel_logits']  # [bsz, num_generated_triples, num_rel+1]
#         idx = self._get_src_permutation_idx(indices)
#         target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)])
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
#         target_classes[idx] = target_classes_o
#         # Apply Focal Loss
#         loss = self.focal_loss(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=self.rel_weight)
#         losses = {'relation': loss}
#         return losses

#     @torch.no_grad()
#     def loss_cardinality(self, outputs, targets, indices):
#         pred_rel_logits = outputs['pred_rel_logits']
#         device = pred_rel_logits.device
#         tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
#         card_pred = (pred_rel_logits.argmax(-1) != pred_rel_logits.shape[-1] - 1).sum(1)
#         card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
#         losses = {'cardinality_error': card_err}
#         return losses

#     def _get_src_permutation_idx(self, indices):
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def _get_tgt_permutation_idx(self, indices):
#         batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx

#     def get_loss(self, loss, outputs, targets, indices, **kwargs):
#         loss_map = {
#             'relation': self.relation_loss,
#             'cardinality': self.loss_cardinality,
#             'entity': self.entity_loss
#         }
#         return loss_map[loss](outputs, targets, indices, **kwargs)

#     def entity_loss(self, outputs, targets, indices):
#         idx = self._get_src_permutation_idx(indices)
#         selected_pred_head_start = outputs["head_start_logits"][idx]
#         selected_pred_head_end = outputs["head_end_logits"][idx]

#         target_head_start = torch.cat([t["head_start_index"][i] for t, (_, i) in zip(targets, indices)])
#         target_head_end = torch.cat([t["head_end_index"][i] for t, (_, i) in zip(targets, indices)])

#         # Apply Focal Loss for entity position predictions
#         head_start_loss = self.focal_loss(selected_pred_head_start, target_head_start)
#         head_end_loss = self.focal_loss(selected_pred_head_end, target_head_end)
#         losses = {'head_entity': 0.5 * (head_start_loss + head_end_loss)}
#         return losses

#     @staticmethod
#     def empty_targets(targets):
#         return all(len(target["relation"]) == 0 for target in targets)









# class SetCriterion(nn.Module):
#     """ This class computes the loss for Set_RE.
#     The process happens in two steps:
#         1) we compute hungarian assignment between ground truth and the outputs of the model
#         2) we supervise each pair of matched ground-truth / prediction (supervise class, subject position and object position)
#     """
#     def __init__(self, num_classes, loss_weight, na_coef, losses, matcher):
#         """ Create the criterion.
#         Parameters:
#             num_classes: number of relation categories
#             matcher: module able to compute a matching between targets and proposals
#             loss_weight: dict containing as key the names of the losses and as values their relative weight.
#             na_coef: list containg the relative classification weight applied to the NA category and positional classification weight applied to the [SEP]
#             losses: list of all the losses to be applied. See get_loss for list of available losses.
#         """
#         super().__init__()
#         self.num_classes = num_classes
#         self.loss_weight = loss_weight
#         self.matcher = HungarianMatcher(loss_weight, matcher)
#         self.losses = losses
#         rel_weight = torch.ones(self.num_classes + 1)
#         rel_weight[-1] = na_coef
#         self.register_buffer('rel_weight', rel_weight)

#     def forward(self, outputs, targets):
#         """ This performs the loss computation.
#         Parameters:
#              outputs: dict of tensors, see the output specification of the model for the format
#              targets: list of dicts, such that len(targets) == batch_size.
#                       The expected keys in each dict depends on the losses applied, see each loss' doc
#         """
#         # Retrieve the matching between the outputs of the last layer and the targets
#         indices = self.matcher(outputs, targets)
#         # Compute all the requested losses
#         losses = {}
#         for loss in self.losses:
#             if loss == "entity" and self.empty_targets(targets):
#                 pass
#             else:
#                 losses.update(self.get_loss(loss, outputs, targets, indices))
#         losses = sum(losses[k] * self.loss_weight[k] for k in losses.keys() if k in self.loss_weight)
#         return losses

#     def relation_loss(self, outputs, targets, indices):
#         src_logits = outputs['pred_rel_logits']  # [bsz, num_generated_triples, num_rel+1]
#         idx = self._get_src_permutation_idx(indices)
#         target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)])
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                     dtype=torch.float32, device=src_logits.device)
#         target_classes[idx] = target_classes_o

#         # Replace cross-entropy with Smooth L1
#         loss = F.smooth_l1_loss(src_logits, target_classes, reduction='mean')
#         losses = {'relation': loss}
#         return losses

#     @torch.no_grad()
#     def loss_cardinality(self, outputs, targets, indices):
#         """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty triples
#         This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
#         """
#         pred_rel_logits = outputs['pred_rel_logits']
#         device = pred_rel_logits.device
#         tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
#         # Count the number of predictions that are NOT "no-object" (which is the last class)
#         card_pred = (pred_rel_logits.argmax(-1) != pred_rel_logits.shape[-1] - 1).sum(1)
#         card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
#         losses = {'cardinality_error': card_err}
#         return losses

#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     def _get_tgt_permutation_idx(self, indices):
#         # permute targets following indices
#         batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx

#     def get_loss(self, loss, outputs, targets, indices,  **kwargs):
#         loss_map = {
#             'relation': self.relation_loss,
#             'cardinality': self.loss_cardinality,
#             'entity': self.entity_loss
#         }
#         return loss_map[loss](outputs, targets, indices, **kwargs)

#     def entity_loss(self, outputs, targets, indices):
#         idx = self._get_src_permutation_idx(indices)
#         selected_pred_head_start = outputs["head_start_logits"][idx]
#         selected_pred_head_end = outputs["head_end_logits"][idx]

#         target_head_start = torch.cat([t["head_start_index"][i] for t, (_, i) in zip(targets, indices)])
#         target_head_end = torch.cat([t["head_end_index"][i] for t, (_, i) in zip(targets, indices)])

#         head_start_loss = F.smooth_l1_loss(selected_pred_head_start, target_head_start, reduction='mean')
#         head_end_loss = F.smooth_l1_loss(selected_pred_head_end, target_head_end, reduction='mean')
#         losses = {'head_entity': 1 / 2 * (head_start_loss + head_end_loss)}
#         return losses

#     @staticmethod
#     def empty_targets(targets):
#         flag = True
#         for target in targets:
#             if len(target["relation"]) != 0:
#                 flag = False
#                 break
#         return flag
