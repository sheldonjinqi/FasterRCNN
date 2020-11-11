import torch
import torch.nn.functional as F
from torch import nn
from utils import *
import torchvision.ops as ops

class BoxHead(torch.nn.Module):
    def __init__(self,device='cuda',Classes=3,P=7):
        #added
        super(BoxHead,self).__init__()

        self.device = device
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
        labels = []
        regressor_target = []
        for i in range(len(gt_labels)):
            total_proposals += len(proposals[i])
            labels_tmp = torch.zeros(len(proposals[i]),1)
            regressor_target_tmp = torch.zeros(len(proposals[i]),4)
            for j in range(len(proposals[i])):
                proposal = proposals[i][j]
                iou = IOU(bbox[i],proposal)
                iou_idx = (iou > 0.5).nonzero()
                #if has iou > 0.5 with more than one gt box
                if len(iou_idx) > 1 :
                    iou_idx = (iou > 0.99* iou.max()).nonzero()
                elif len(iou_idx) == 0:
                    continue
                labels_tmp[j] = gt_labels[i][iou_idx]
                regressor_target_tmp[j] = bbox[i][iou_idx]
            # print(labels_tmp)
            labels.extend(labels_tmp)
            regressor_target.extend(regressor_target_tmp)
        labels = torch.stack(labels)
        regressor_target = torch.stack(regressor_target)

        x1,y1,x2,y2 = regressor_target.T
        x_gt = 0.5 * (x1+x2)
        y_gt = 0.5 * (y1+y2)
        w_gt = abs(x1-x2)
        h_gt = abs(y1-y2)
        xp = 0.5*(torch.cat([x1[:,0] for x1 in proposals]) + torch.cat([x1[:,2] for x1 in proposals]))
        yp = 0.5*(torch.cat([x1[:,1] for x1 in proposals]) + torch.cat([x1[:,3] for x1 in proposals]))
        wp = torch.abs(torch.cat([x1[:,0] for x1 in proposals]) - torch.cat([x1[:,2] for x1 in proposals]))
        hp = torch.abs((torch.cat([x1[:,1] for x1 in proposals]) - torch.cat([x1[:,3] for x1 in proposals])))

        tx = (x_gt-xp)/wp
        ty = (y_gt-yp)/hp
        tw = torch.log(w_gt/wp)
        th = torch.log(h_gt/hp)

        regressor_target = torch.stack((tx,ty,tw,th),1)
        regressor_target[w_gt==0] = 0
        return labels,regressor_target



    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    # def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=7):
    #     #####################################
    #     # Here you can use torchvision.ops.RoIAlign check the docs
    #     #####################################
    #
    #     #using for loop first, try vectorizing to speed up
    #     feature_vectors = []
    #     scale_list = [[] for i in range(len(fpn_feat_list))] #len(FPN), each sublist contains all proposasls in that scale
    #     total_proposals = 0
    #     for proposal in proposals:
    #         x1,y1,x2,y2 = proposal.T
    #         p_width = np.abs(x1-x2)
    #         p_height =np.abs(y1-y2)
    #         fpn_idx = (4+np.log2(np.sqrt(p_width*p_height)/224)).floor().clamp(min=2,max=5) #k-values are clipped to [2,5] according to piazza
    #         fpn_idx -= 2
    #         for i in range(len(fpn_feat_list)):
    #             matched_proposals = proposal[fpn_idx == i]
    #             # convert proposal box from img coord to featuremap coord
    #             # matched_proposals *= fpn_feat_list[i].shape[-1]/ 1088
    #             matched_proposals[:, (0,2)] *= fpn_feat_list[i].shape[-1] / 1088 # rescaling the x-coords
    #             matched_proposals[:, (1,3)] *= fpn_feat_list[i].shape[-2] / 800 # rescaling the y-coords
    #             scale_list[i].append(matched_proposals)
    #
    #         total_proposals += len(proposal)
    #
    #     for i in range(len(fpn_feat_list)):
    #         featmap = fpn_feat_list[i]
    #         #convert proposal box from img coord to featuremap coord
    #         proposals_fp = scale_list[i]  #list:len(bz){per_image_proposals,4}
    #         aligned_featmap = ops.roi_align(featmap,boxes=proposals_fp,output_size=P) #shape: (total_proposals in one feature level,256,P,P)
    #         feature_vectors_tmp = torch.flatten(aligned_featmap, -3, -1)  # shape: (total_proposals in one feature level,256*P^2)
    #         feature_vectors.append(feature_vectors_tmp)
    #     feature_vectors = torch.cat(feature_vectors,dim=0)  # shape: (total_proposals,256*P^2)
    #     assert feature_vectors.shape[0] == total_proposals
    #
    #     return feature_vectors

    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    #loop through one by one
    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=7):
        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################

        #using for loop first, try vectorizing to speed up
        feature_vectors = []
        scale_list = [[] for i in range(len(fpn_feat_list))] #len(FPN), each sublist contains all proposasls in that scale
        total_proposals = 0
        for i in range(len(proposals)):
            for box in proposals[i]:
                x1,y1,x2,y2 = box
                p_width = np.abs(x1 - x2)
                p_height = np.abs(y1 - y2)
                k = (4 + np.log2(np.sqrt(p_width * p_height) / 224)).floor().clamp(min=2,
                                                                                         max=5).int()  # k-values are clipped to [2,5] according to piazza
                k -= 2
                x1 *= fpn_feat_list[k].shape[-1] / 1088
                x2 *= fpn_feat_list[k].shape[-1] / 1088
                y1 *= fpn_feat_list[k].shape[-2] / 800
                y2 *= fpn_feat_list[k].shape[-2] / 800

                # proposal_box = [torch.tensor([[x1,y1,x2,y2]])]
                proposal_box = torch.tensor([[i,x1,y1,x2,y2]])
                aligned_box = ops.roi_align(fpn_feat_list[k],boxes=proposal_box,output_size=P)
                feature_vectors.append(aligned_box)
        feature_vectors = torch.cat(feature_vectors,dim=0)  # shape: (total_proposals,256*P^2)
        feature_vectors = torch.flatten(feature_vectors, -3, -1)

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


    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self,p_out,n_out,p_out_gt):

        # TODO compute classifier's loss
        # p_out_gt = torch.ones(len(p_out),1)
        n_out_gt = torch.zeros(len(n_out)).to(self.device)
        CE_loss = nn.CrossEntropyLoss()

        loss_p = CE_loss(p_out,p_out_gt.long())*len(p_out_gt)
        loss_n = CE_loss(n_out,n_out_gt.long())*len(n_out_gt)

        loss = (loss_p + loss_n) / (len(p_out_gt) + len(n_out_gt))

        return loss



    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self,pos_target_coord,pos_out_r,positive_gt_label, effective_batch=120):
            #torch.nn.SmoothL1Loss()
            # TODO compute regressor's loss
            # TODO can be vectorized
            SmoothL1Loss = nn.SmoothL1Loss(reduction='sum')
            loss = 0
            for i in range(len(pos_target_coord)):
                c_idx = positive_gt_label[i].int() -1
                loss_temp = SmoothL1Loss(pos_out_r[i][c_idx*4:c_idx*4+4],pos_target_coord[i])
                loss += loss_temp
            print('effective batch',effective_batch)
            loss /=   effective_batch

            return loss


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

        #testing data
        # loss_data = torch.load('./Test_Cases/Loss/loss_test6.pt')
        # random_permutation_background = loss_data.get('random_permutation_background')
        # random_permutation_foreground = loss_data.get('random_permutation_foreground')
        # loss_reg = loss_data.get('loss_reg')
        # loss_clas = loss_data.get('loss_clas')

        # Pick max(M/2,available positive ground truth anchors)
        positive_gt_anchor_idx = torch.where(labels != 0) # all proposals that are not background are positive
        negative_gt_anchor_idx = torch.where(labels == 0) # Only background proposals are negtive



        num_pos_anchor = int(min(effective_batch *0.5, len(positive_gt_anchor_idx[0]))) # 0.75 used for 3:1 ratio
        num_neg_anchor = effective_batch - num_pos_anchor

        # create random index for pos and negative gt anchor, and pick corresponding # of idx
        pos_rand_idx = torch.randperm(len(positive_gt_anchor_idx[0]))[:num_pos_anchor]
        neg_rand_idx = torch.randperm(len(negative_gt_anchor_idx[0]))[:num_neg_anchor]

        p_classifier_out = class_logits[positive_gt_anchor_idx]  # shape: (num_pos_anchor,C+1)
        p_classifier_out = p_classifier_out[pos_rand_idx]
        n_classifier_out = class_logits[negative_gt_anchor_idx]  # shape: (effective_batch - num_pos_anchor,C+1)
        n_classifier_out = n_classifier_out[neg_rand_idx]

        p_regressor_gt = regression_targets[positive_gt_anchor_idx[0]]  # shape: (num_pos_anchor,4)
        p_regressor_gt = p_regressor_gt[pos_rand_idx]
        p_regressor_pred = box_preds[positive_gt_anchor_idx[0]]  # shape: (num_pos_anchor,4*C)
        p_regressor_pred = p_regressor_pred[pos_rand_idx]

        positive_gt_label = labels[positive_gt_anchor_idx]  # stores all positive labels
        positive_gt_label = positive_gt_label[pos_rand_idx] #corresponding labels for each proposal

        # print('random_permutation_background', random_permutation_background)
        # print('random_permutation_foreground', random_permutation_foreground)
        # print('p_idx',pos_rand_idx)
        # print('n_idx',neg_rand_idx)

        #using test data:
        # foreground_ind = (labels > 0).nonzero()
        # background_ind = (labels ==0).nonzero()
        # foreground_target = regression_targets[foreground_ind[random_permutation_foreground]].squeeze()
        # foreground_prediction = box_preds[foreground_ind[random_permutation_foreground]].squeeze()
        # foreground_label = labels[foreground_ind[random_permutation_foreground]].squeeze()
        # effective_batch = loss_data.get('effective_batch')


        # loss_class = self.loss_class(class_logits[foreground_ind[random_permutation_foreground]].squeeze().to(self.device), class_logits[background_ind[random_permutation_background]].squeeze().to(self.device), \
        #                              foreground_label.to(self.device))
        # loss_regr  =  self.loss_reg(foreground_target.to(self.device), foreground_prediction.to(self.device), \
        #                             foreground_label.to(self.device),effective_batch=effective_batch)
        #
        # print('calculated class',loss_class)
        # print('calculated regr', loss_regr)
        # print('gt_loss_clas', loss_clas)
        # print('gt_loss_reg', loss_reg)


        loss_class = self.loss_class(p_classifier_out.to(self.device), n_classifier_out.to(self.device),positive_gt_label.to(self.device))
        loss_regr  =  self.loss_reg(p_regressor_gt.to(self.device), p_regressor_pred.to(self.device),positive_gt_label.to(self.device),effective_batch=effective_batch)

        l = 10 # randomly set to 10
        loss = loss_class + l*loss_regr

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
    print('hi')