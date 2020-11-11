import torch
import numpy as np
from BoxHead import *
from utils import *

box_head = BoxHead()

# Test for Ground Truth
#mis one proposal for last test case due to IOU numerical precision
# for i in range(7):
#     gt_data = torch.load('./Test_Cases/GroundTruth/ground_truth_test'+str(i)+'.pt')
#     # print(gt_data.keys())
#
#     proposals = gt_data.get('proposals')
#     gt_labels = gt_data.get('gt_labels')
#     bbox = gt_data.get('bbox')
#     test_labels = gt_data.get('labels')
#     test_regressor_target = gt_data.get('regressor_target')
#
#     labels, regressor_target = box_head.create_ground_truth(proposals,gt_labels,bbox)
#
#     labels_check = torch.sum(test_labels-labels)
#     # print(bbox[1])
#     # print(proposals[1][13])
#     # print(type(IOU(bbox[1],proposals[1][13])))
#     # print('test_labels:\n',torch.where(test_labels != 0 ))
#     # print('labels:\n',torch.where(labels != 0 ))
#     #only checks the proposal boxes that are not assigned to background according to README
#     regressor_check = torch.sum(test_regressor_target[torch.where(test_labels != 0 )[0]] - \
#                                 regressor_target[torch.where(labels != 0 )[0]])
#
#     print('labels_check:',labels_check)
#     print('regressor_check',regressor_check)
# print(test_regressor_target[torch.where(test_labels != 0 )[0]])
# print(regressor_target[torch.where(labels != 0 )[0]])
# print(torch.sum(test_regressor_target[torch.where(test_labels != 0 )[0]]-regressor_target[torch.where(labels != 0 )[0]]))
# print(test_regressor_target[:10])
# print(regressor_target[:10])

# Test for Loss
for i in range(7):
    loss_data = torch.load('./Test_Cases/Loss/loss_test'+str(i)+'.pt')
    random_permutation_background = loss_data.get('random_permutation_background')
    random_permutation_foreground = loss_data.get('random_permutation_foreground')
    clas_logits = loss_data.get('clas_logits')
    box_preds = loss_data.get('box_preds')
    labels = loss_data.get('labels')
    regression_targets = loss_data.get('regression_targets')
    effective_batch = loss_data.get('effective_batch')
    loss_reg = loss_data.get('loss_reg')
    loss_clas = loss_data.get('loss_clas')

    p_loss = box_head.compute_loss(clas_logits,box_preds,labels,regression_targets,l=1,effective_batch=effective_batch)
    # print('random_permutation_background',random_permutation_background.shape)
    # print('random_permutation_foreground',random_permutation_foreground.shape)
    print('calculated loss',p_loss)
    print('loss_sum', loss_reg+loss_clas)
    print('loss_reg',loss_reg)
    print('loss_clas',loss_clas)
# Test for Multi-Scale ROI align

# for i in range(4):
#     msra_data = torch.load('./Test_Cases/MultiScaleRoiAlign/multiscale_RoIAlign_test'+str(i)+'.pt')
#
#     fpn_feat_list = msra_data.get('fpn_feat_list')
#     proposals = msra_data.get('proposals')
#     output_feature_vectors = msra_data.get('output_feature_vectors')
#     feature_vectors = box_head.MultiScaleRoiAlign(fpn_feat_list,proposals)
#     roi_check = output_feature_vectors - feature_vectors
#
#     if torch.sum(roi_check) > 0:
#         print('test failed at dataset ',str(i))
#         exit()
# print('passed roi test')
