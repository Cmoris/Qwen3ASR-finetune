import torch

def get_rope_index(self, attention_mask=None):
    """
    Make every two valid tokens share the same position:
    valid token index: 0,1,2,3,4,5
    position id:       0,0,1,1,2,2
    """
    valid_pos = attention_mask.long().cumsum(-1) - 1
    valid_pos.masked_fill_(attention_mask == 0, 1)

    position_ids_2d = valid_pos // 2

    position_ids = position_ids_2d.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)

    max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
    mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

    return position_ids, mrope_position_deltas