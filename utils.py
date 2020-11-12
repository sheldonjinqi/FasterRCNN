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
        lefts = center_xs - ws / 2
        rights = center_xs + ws / 2
        uppers = center_ys - hs / 2
        bottoms = center_ys + hs / 2

    else:
        lefts, uppers, rights, bottoms = boxA.T
        ws = abs(rights-lefts)
        hs = abs(uppers-bottoms)
    target_left, target_upper, target_right, target_bottom = boxB.T

    areas1 = ws * hs  # torch.size(grid_size[0], grid_size[1])
    area2 = abs(target_right - target_left) * abs(target_upper - target_bottom)  # float


    # max(a,b)
    x_overlap = (torch.min(rights, target_right) - torch.max(lefts, target_left)).clamp(min=0)
    y_overlap = (torch.min(bottoms, target_bottom) - torch.max(uppers, target_upper)).clamp(min=0)
    overlapArea = x_overlap * y_overlap

    iou = (overlapArea + 0.001) / (areas1 + area2 - overlapArea + 0.001)


    return iou

# iou function copied from yolo implementation, calculate iou between two groups of boxes
#input shape: {(per_image_proposals,4)} ([x1,y1,x2,y2] format)
#             {(n_obj, 4)}
def iou_entire(proposal, bbox):
    # if torch.is_tensor(net_out):
    #     net_out = net_out.cpu().detach()
    # if torch.is_tensor(target):
    #     target = target.cpu().detach()

    p_x1, p_y1, p_x2, p_y2 = proposal.T
    g_x1, g_y1, g_x2, g_y2 = bbox.T

    p_x1_mesh, g_x1_mesh = np.meshgrid(p_x1,g_x1)
    p_x2_mesh, g_x2_mesh = np.meshgrid(p_x2, g_x2)
    p_y1_mesh, g_y1_mesh = np.meshgrid(p_y1, g_y1)
    p_y2_mesh, g_y2_mesh = np.meshgrid(p_y2, g_y2)

    p_w = p_x2_mesh - p_x1_mesh
    p_h = p_y1_mesh - p_y2_mesh
    g_w = g_x2_mesh - g_x1_mesh
    g_h = g_y1_mesh - g_y2_mesh



    overlap_w = np.minimum(p_x2_mesh, g_x2_mesh) - np.maximum(p_x1_mesh, g_x1_mesh)
    overlap_h = np.minimum(p_y2_mesh, g_y2_mesh) - np.maximum(p_y1_mesh, g_y1_mesh)

    a_overlap = overlap_w * overlap_h
    # edgecase no prediction box or no overlapping

    a_overlap[np.where(overlap_w <= 0)] = 0
    a_overlap[np.where(overlap_h <= 0)] = 0

    # pdb.set_trace()
    a_union = (p_w) * (p_h) + (g_w) * (g_h) - a_overlap
    # print('area of union',a_union)
    # print('area of overlap', a_overlap)
    iou = a_overlap / a_union
    # pdb.set_trace()
    # print('iou',iou.shape)
    return iou

# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)

#TODO complete this function
# regressed box is in tx, ty, tw, th
# proposals are in x1,y1,x2,y2
def output_decoding(regressed_boxes_t,flatten_proposals, device='cpu'):
    x_p = 0.5*(flatten_proposals[:,0]+flatten_proposals[:,2])
    y_p = 0.5*(flatten_proposals[:,1]+flatten_proposals[:,3])
    w_p = torch.abs(flatten_proposals[:,2] - flatten_proposals[:,0])
    h_p = torch.abs(flatten_proposals[:,3] - flatten_proposals[:,1])
    x = regressed_boxes_t[:,0] * w_p + x_p
    y = regressed_boxes_t[:,1] * h_p + y_p
    w = torch.exp(regressed_boxes_t[:,2]) * w_p
    h = torch.exp(regressed_boxes_t[:,3]) * h_p

    box = torch.zeros_like(flatten_proposals)
    box[:, 0] = x - w * 0.5
    box[:, 1] = y - h * 0.5
    box[:, 2] = w
    box[:, 3] = h

    return box