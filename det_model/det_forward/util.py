import torch


# Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2`` in batch.
# boxes1 (torch.Tensor): batch of List of bounding boxes
# boxes2 (torch.Tensor): batch of List of bounding boxes
#
# Note:
#     List format: [[xc, yc, w, h],...] * batch_size
def batch_cxywh_ious(boxes1, boxes2):
    b1x1, b1y1 = (boxes1[:, :, :2] - (boxes1[:, :, 2:4] / 2)).split(1, dim=2)
    b1x2, b1y2 = (boxes1[:, :, :2] + (boxes1[:, :, 2:4] / 2)).split(1, dim=2)
    b2x1, b2y1 = (boxes2[:, :, :2] - (boxes2[:, :, 2:4] / 2)).split(1, dim=2)
    b2x2, b2y2 = (boxes2[:, :, :2] + (boxes2[:, :, 2:4] / 2)).split(1, dim=2)

    dx = (b1x2.min(b2x2.permute(0, 2, 1)) - b1x1.max(b2x1.permute(0, 2, 1))).clamp(min=0)
    dy = (b1y2.min(b2y2.permute(0, 2, 1)) - b1y1.max(b2y1.permute(0, 2, 1))).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)

    unions = (areas1 + areas2.permute(0, 2, 1)) - intersections

    return intersections / (unions + 1e-10)


def iou_xywh(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:

    # return N x M
    # N = box1.size(0)
    # M = box2.size(0)

    lt = torch.max(box1[:, :2].view(-1, 1, 2),  # N x 1 x 2
                   box2[:, :2].view(1, -1, 2))  # 1 x M x 2

    rb = torch.min((box1[:, :2] + box1[:, 2:]).view(-1, 1, 2),
                   (box2[:, :2] + box2[:, 2:]).view(1, -1, 2))

    intersect_wh = (rb - lt).clamp(min=0)

    intersect = intersect_wh[:, :, 0] * intersect_wh[:, :, 1]

    area1 = (box1[:, 2] * box1[:, 3]).view(-1, 1)  # N x 1
    area2 = (box2[:, 2] * box2[:, 3]).view(1, -1)  # 1 x M

    return intersect / (area1 + area2 - intersect + 1e-10)


# nms(non maximum suppression) and soft nms implement
# `boxes`: input (num_box, 4) shape torch.Tensor in xywh format
# `scores`: input confidence bound to boxes
# `mode`: 'binary' as nms, 'linear' as soft nms using linear iou weight penalty,
#           'gaussian' as  soft nms using gaussian iou weight penalty
# `overlap`: iou threshould to suppress boxes in 'binary' mode or use penalty in 'linear' mode
# `sigma`: sigma value in 'gaussian' mode
# `drop_threshould`: the suppress threshould on scores
# `topk`: the maximum number of result boxes
def nms(boxes: torch.Tensor, scores: torch.Tensor, mode: str = 'binary',
        overlap: float = 0.5, sigma: float = 0.5, drop_threshould: float = 0.001,
        topk: int = 200):
    assert mode in ['linear', 'gaussian', 'binary'], \
        "nms method should be 'linear', 'gaussian' or 'binary'"
    assert scores.size(0) == boxes.size(0) and boxes.size(0) != 0

    num_box = boxes.size(0)
    boxes_ = boxes.clone()
    scores_ = scores.clone()
    ori_scores_ = scores.clone()

    keep_boxes = []
    keep_scores = []
    for _ in range(min(num_box, topk)):
        if len(boxes_) == 0:
            break
        # get max score box, add to keep
        max_score_id = scores_.argmax()
        max_ori_score = ori_scores_[max_score_id].view(1).clone()
        max_box = boxes_[max_score_id, :].view(1, 4).clone()
        keep_boxes.append(max_box)
        keep_scores.append(max_ori_score)

        # swap first box to max box index, get left boxes
        scores_[max_score_id] = scores_[0]
        ori_scores_[max_score_id] = ori_scores_[0]
        boxes_[max_score_id, :] = boxes_[0, :]
        boxes_ = boxes_[1:, :]
        scores_ = scores_[1:]
        ori_scores_ = ori_scores_[1:]

        # calculate ious between max box and left boxes
        ious = iou_xywh(max_box, boxes_).squeeze()

        weights = torch.ones_like(ious)
        if mode == 'linear':
            mask = ious > overlap
            weights[mask] = 1 - ious[mask]
        elif mode == 'gaussian':
            weights = (-(ious * ious) * (1.0 / sigma)).exp()
        else:  # binary
            mask = ious > overlap
            weights[mask] = 0
        scores_ = weights * scores_
        left_id = scores_ > drop_threshould
        boxes_ = boxes_[left_id, :]
        scores_ = scores_[left_id]
        ori_scores_ = ori_scores_[left_id]

    return torch.cat(keep_boxes), torch.cat(keep_scores)
