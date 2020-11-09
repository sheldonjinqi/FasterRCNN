import numpy as np
import torch
from functools import partial
def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function compute the IOU between two set of boxes 
def IOU(boxA, boxB):
    ##################################
    # TODO compute the IOU between the boxA, boxB boxes
    ##################################

    '''
      :param: boxA: all anchors in a grid map torch.size((grid_size[0],grid_size[1],4)) , (x/col, y/row, w, h)
      :param: boxB: single ground truth box torch.Tensor([x1,y1,x2,y2]
    '''

    if len(boxA.size()) == 3:
        center_xs, center_ys, ws, hs = boxA[:, :, 0], boxA[:, :, 1], boxA[:, :, 2], boxA[:, :, 3]
    else:
        center_xs, center_ys, ws, hs = boxA[:, 0], boxA[:, 1], boxA[:, 2], boxA[:, 3]

    target_left, target_upper, target_right, target_bottom = boxB

    areas1 = ws * hs  # torch.size(grid_size[0], grid_size[1])
    area2 = abs(target_right - target_left) * abs(target_upper - target_bottom)  # float

    lefts = center_xs - ws / 2
    rights = center_xs + ws / 2
    uppers = center_ys - hs / 2
    bottoms = center_ys + hs / 2

    # max(a,b)
    x_overlap = (torch.min(rights, target_right) - torch.max(lefts, target_left)).clamp(min=0)
    y_overlap = (torch.min(bottoms, target_bottom) - torch.max(uppers, target_upper)).clamp(min=0)
    overlapArea = x_overlap * y_overlap

    iou = (overlapArea + 0.001) / (areas1 + area2 - overlapArea + 0.001)


    return iou


# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)
def output_decodingd(regressed_boxes_t,flatten_proposals, device='cpu'):
    
    return box