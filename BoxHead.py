import torch
import torch.nn.functional as F
from torch import nn
from utils import *
import torchvision.ops as ops

class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7):
        self.C=Classes
        self.P=P
        # TODO initialize BoxHead

        self.intermediate = nn.Sequential(
            nn.Linear(256*(self.P ** 2),1024),
            nn.Linear(1024,1024),
            nn.ReLU()
        )
        self.classfier_head = nn.Sequential(
            nn.Linear(1024,self.C+1),
            nn.Softmax()
        )


        self.regressor_head = nn.Sequential(
            nn.Linear(1024, 4*self.C)
        )

    #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
    def create_ground_truth(self,proposals,gt_labels,bbox):
        total_proposals = 0
        for i in range(len(gt_labels)):
            total_proposals += len(proposals[i])
            labels_tmp = torch.zeros(len(proposals[i]),1)
            regressor_target_tmp = torch.zeros(len(proposals[i]),4)
            for j in range(len(proposals[i])):
                proposal = proposals[j]
                iou = IOU(bbox,proposal)
                iou_idx = (iou > 0.5).nonzero()
                #if has iou > 0.5 with more than one gt box
                if len(iou_idx) > 1 :
                    iou_idx = (iou > 0.99* iou.max()).nonzero()
                labels_tmp[j] = gt_labels[i][iou_idx]
                regressor_target_tmp[j] = bbox[i][iou_idx]

        pass
        return labels,regressor_target



    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=7):
        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################

        #using for loop first, try vectorizing to speed up
        feature_vectors = []
        scale_list = [[] for i in range(len(fpn_feat_list))] #len(FPN), each sublist contains all proposasls in that scale
        for proposal in proposals:
            x1,y1,x2,y2 = proposal
            p_width = np.abs(x1-x2)
            p_height =np.abs(y1-y2)
            fpn_idx = 4+np.log2(np.sqrt(p_width*p_height)/224) #has a range from 2 to 5
            fpn_idx -= 2
            for i in range(len(fpn_feat_list)):
                scale_list[i].append(proposal[fpn_idx == i])

        for i in range(len(fpn_feat_list)):
            featmap = fpn_feat_list[i]
            #convert proposal box from img coord to featuremap coord
            proposals_fp = scale_list[i] *  featmap.shape[-1] / 1088 #list:len(bz){per_image_proposals,4}
            aligned_featmap = ops.roi_align(featmap,boxes=proposals_fp,output_size=P) #shape: (total_proposals,256,P,P)
            feature_vectors = torch.flatten(aligned_featmap, -2, -1)  # shape: (total_proposals,256,P^2)

        total_proposals = 0
        total_proposals = [total_proposals + p.shape[0] for p in proposals]
        assert aligned_featmap.shape[0] == total_proposals

        return feature_vectors

    # This function does the post processing for the results of the Box Head for a batch of images
    # Use the proposals to distinguish the outputs from each image
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
    #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
    #       conf_thresh: scalar
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50):

        return boxes, scores, labels




    # Compute the total loss of the classifier and the regressor
    # Input:
    #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
    #      box_preds: (total_proposals,4*C)      (as outputed from forward)
    #      labels: (total_proposals,1)
    #      regression_targets: (total_proposals,4)
    #      l: scalar (weighting of the two losses)
    #      effective_batch: scalar
    # Outpus:
    #      loss: scalar
    #      loss_class: scalar
    #      loss_regr: scalar
    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=1,effective_batch=150):

        return loss, loss_class, loss_regr



    # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
    def forward(self, feature_vectors):
        feature_vectors = self.intermediate(feature_vectors)
        class_logits = self.classfier_head(feature_vectors)
        box_pred = self.regressor_head(feature_vectors)
        return class_logits, box_pred

if __name__ == '__main__':
