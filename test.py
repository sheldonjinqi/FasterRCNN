import torch
import numpy as np
from BoxHead import *
from utils import *

box_head = BoxHead()

# Test for Ground Truth
#mis one proposal for last test case due to IOU numerical precision
for i in range(7):
    gt_data = torch.load('./Test_Cases/GroundTruth/ground_truth_test'+str(i)+'.pt')
    # print(gt_data.keys())

    proposals = gt_data.get('proposals')
    gt_labels = gt_data.get('gt_labels')
    bbox = gt_data.get('bbox')
    test_labels = gt_data.get('labels')
    test_regressor_target = gt_data.get('regressor_target')

    labels, regressor_target = box_head.create_ground_truth(proposals,gt_labels,bbox)

    labels_check = torch.sum(test_labels-labels)
    # print(bbox[1])
    # print(proposals[1][13])
    # print(type(IOU(bbox[1],proposals[1][13])))
    # print('test_labels:\n',torch.where(test_labels != 0 ))
    # print('labels:\n',torch.where(labels != 0 ))
    #only checks the proposal boxes that are not assigned to background according to README
    regressor_check = torch.sum(test_regressor_target[torch.where(test_labels != 0 )[0]] - \
                                regressor_target[torch.where(labels != 0 )[0]])

    print('labels_check:',labels_check)
    print('regressor_check',regressor_check)
# print(test_regressor_target[torch.where(test_labels != 0 )[0]])
# print(regressor_target[torch.where(labels != 0 )[0]])
# print(torch.sum(test_regressor_target[torch.where(test_labels != 0 )[0]]-regressor_target[torch.where(labels != 0 )[0]]))
# print(test_regressor_target[:10])
# print(regressor_target[:10])

# Test for Loss
loss_data = torch.load('./Test_Cases/Loss/loss_test0.pt')
# print(loss_data.keys())

# Test for Multi-Scale ROI align

for i in range(4):
    msra_data = torch.load('./Test_Cases/MultiScaleRoiAlign/multiscale_RoIAlign_test'+str(i)+'.pt')

    fpn_feat_list = msra_data.get('fpn_feat_list')
    proposals = msra_data.get('proposals')
    output_feature_vectors = msra_data.get('output_feature_vectors')
    feature_vectors = box_head.MultiScaleRoiAlign(fpn_feat_list,proposals)
    roi_check = output_feature_vectors - feature_vectors

    if torch.sum(roi_check) > 0:
        print('test failed at dataset ',str(i))
        exit()
print('passed roi test')
