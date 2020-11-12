import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
from utils import *

import torchvision


class RPNHead(torch.nn.Module):

    def __init__(self,  device='cuda', anchors_param=dict(ratio=0.8,scale= 256, grid_size=(50, 68), stride=16)):
        # Initialize the backbone, intermediate layer clasifier and regressor heads of the RPN
        super(RPNHead,self).__init__()

        self.device=device
        # TODO Define Backbone
        self.conv_backbone = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=5,stride=1 ,padding=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
            nn.Conv2d(16,32,kernel_size=5,stride=1, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2), nn.BatchNorm2d(256), nn.ReLU(),
        )

        # TODO  Define Intermediate Layer
        self.intermediate = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # TODO  Define Proposal Classifier Head
        # input dimension: 256,Sy,Sx
        # output dimenstion: 1,Sy,Sx
        self.classfier_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # TODO Define Proposal Regressor Head
        # input dimension: 256,Sy,Sx
        # output dimenstion: 4,Sy,Sx
        self.regressor_head = nn.Sequential(
            nn.Conv2d(256, 4, kernel_size=1, stride=1, padding=0)
        )

        #  find anchors
        self.anchors_param=anchors_param
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict={}

    # Forward  the input through the backbone the intermediate layer and the RPN heads
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       logits: (bz,1,grid_size[0],grid_size[1])}
    #       bbox_regs: (bz,4, grid_size[0],grid_size[1])}
    def forward(self, X):

        #TODO forward through the Backbone
        X = self.conv_backbone(X)

        #TODO forward through the Intermediate layer
        X = self.intermediate(X)

        #TODO forward through the Classifier Head
        logits = self.classfier_head(X)

        #TODO forward through the Regressor Head
        bbox_regs = self.regressor_head(X)

        assert logits.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert bbox_regs.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return logits, bbox_regs




    # Forward input batch through the backbone
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       X: (bz,256,grid_size[0],grid_size[1])
    def forward_backbone(self,X):
        #####################################
        # TODO forward through the backbone
        #####################################
        X = self.conv_backbone(X)

        assert X.shape[1:4]==(256,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return X



    # This function creates the anchor boxes
    # Output:
    #       anchors: (grid_size[0],grid_size[1],4)
    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        '''
        :param aspect_ratio: float = w/h
        :param scale: float = sqrt(wh)
        :param grid_sizes: tuple=(50, 68)  -> the shape of feature map & Sx, Sy
        :param stride: int = 16 -> the dimension of the grid cells
        :return: anchors : (grid_size[0],grid_size[1],4) -> (x,y,w,h)
        '''
        ######################################
        # TODO create anchors
        ######################################
        # calculate w,h
        h = (scale**2/aspect_ratio)**0.5
        w = aspect_ratio * h

        # initialize anchors with size((grid_size[0],grid_size[1],4))
        anchors = torch.zeros([grid_sizes[0],grid_sizes[1],4])

        # fill anchor box shape
        anchors[:,:,2:] = torch.Tensor([w,h])

        # fill anchor pos info
        col_list = torch.linspace(8,1080,68)
        row_list = torch.linspace(8,792, 50)

        row_list, col_list= torch.meshgrid(row_list,col_list)

        #  might need to be flipped here
        anchors[:,:,0] = col_list
        anchors[:,:,1] = row_list

        # check flag
        assert anchors.shape == (grid_sizes[0] , grid_sizes[1],4)

        return anchors

    def get_anchors(self):
        return self.anchors



    # This function creates the ground truth for a batch of images by using
    # create_ground_truth internally
    # Input:
    #      bboxes_list: list:len(bz){(n_obj,4)}
    #      indexes:      list:len(bz)
    #      image_shape:  tuple:len(2)
    # Output:
    #      ground_clas: (bz,1,grid_size[0],grid_size[1])
    #      ground_coord: (bz,4,grid_size[0],grid_size[1])
    def create_batch_truth(self,bboxes_list,indexes,image_shape):
        #####################################
        # TODO create ground truth for a batch of images
        #####################################
        bz = len(bboxes_list)
        grid_size = self.anchors_param['grid_size']

        # initialize ground_clas and ground_coord
        ground_clas = torch.ones((bz, 1, grid_size[0], grid_size[1]))
        ground_coord = torch.ones((bz, 4, grid_size[0], grid_size[1]))

        # for bboxes, ind in zip(bboxes_list, indexes):
        for i in range(bz):
            bboxes = bboxes_list[i]
            ind = indexes[i]
            ground_clas[i,:], ground_coord[i,:] = self.create_ground_truth(bboxes, ind, grid_size, self.anchors, image_shape)

        assert ground_clas.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert ground_coord.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return ground_clas, ground_coord


    # This function creates the ground truth for one image
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset used for caching)
    #       grid_size:   tuple:len(2)
    #       anchors:     (grid_size[0],grid_size[1],4)
    # Output:
    #       ground_clas:  (1,grid_size[0],grid_size[1])
    #       ground_coord: (4,grid_size[0],grid_size[1])
    def create_ground_truth(self, bboxes, index, grid_size, anchors, image_size):
        # anchors : (grid_size[0],grid_size[1],4) -> (x,y,w,h)

        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        #####################################################
        # TODO create ground truth for a single image
        #####################################################
        ground_clas = torch.ones((1, grid_size[0], grid_size[1])) * -1
        ground_coord = torch.ones((4, grid_size[0], grid_size[1])) * 0

        for bbox in bboxes:
            # calculate iou for all anchors
            iou = IOU(anchors,bbox) # iou: torch.size(grid_size[0],grid_size[1])

            ## positive labels
            # highest IOU with a ground truth box should be 1
            max_ind = (iou > 0.99* iou.max()).nonzero()

            # anchors with IOU > 0.7 with ANY ground truth box should be 1
            valid_iou_ind = (iou > 0.7).nonzero()

            positive_ind = torch.cat((max_ind, valid_iou_ind), 0)
            ground_clas[0, positive_ind[:, 0], positive_ind[:, 1]] = 1

            # t∗x = (x∗ − xa)/wa t∗y = (y∗ − ya)/ha t∗w = log(w∗/wa) t∗h = log(h∗/ha)
            x1, y1, x2, y2 = bbox
            gt_c_x, gt_c_y, gt_w, gt_h = 0.5 * (x1 + x2), 0.5 * (y1 + y2), abs(x1 - x2), abs(y1 - y2)
            xa, ya, wa, ha = anchors[:, :, 0], anchors[:, :, 1], anchors[:, :, 2], anchors[:, :, 3]
            t_x, t_y, t_w, t_h = (gt_c_x - xa)/wa, (gt_c_y-ya)/ha, torch.log(gt_w/wa), torch.log(gt_h/ha)

            # ground_coord[:, max_ind[:,0],max_ind[:,1]] = torch.Tensor([c_x, c_y, w, h])
            ground_coord[0, positive_ind[:,0],positive_ind[:,1]] = t_x[positive_ind[:,0],positive_ind[:,1]]
            ground_coord[1, positive_ind[:,0],positive_ind[:,1]] = t_y[positive_ind[:,0],positive_ind[:,1]]
            ground_coord[2, positive_ind[:,0],positive_ind[:,1]] = t_w[positive_ind[:,0],positive_ind[:,1]]
            ground_coord[3, positive_ind[:,0],positive_ind[:,1]] = t_h[positive_ind[:,0],positive_ind[:,1]]
            # ground_coord[:, valid_iou_ind[:,0], valid_iou_ind[:,1]] = torch.Tensor([c_x, c_y, w, h]).view(4,1)

            ## negtive labels
            # anchors that are non-positive and have IOU < 0.3 with EVERY ground truth box
            neg_ind = (torch.bitwise_and(iou<0.3, (ground_clas[0]<1))).nonzero()
            ground_clas[0, neg_ind[:,0], neg_ind[:,1]] = 0

        # remove corss-boundry anchors
        center_xs, center_ys, ws, hs = anchors[:, :, 0], anchors[:, :, 1], anchors[:, :, 2], anchors[:, :, 3]
        lefts,rights, uppers, bottoms = center_xs - ws / 2, center_xs + ws / 2, center_ys - hs / 2, center_ys + hs / 2

        h, w = image_size
        cross_bound_w = torch.bitwise_or(lefts < 0, rights >= w)
        cross_bound_h = torch.bitwise_or(uppers< 0 , bottoms >= h)
        cross_bound = torch.bitwise_or(cross_bound_w, cross_bound_h).nonzero()
        ground_clas[0, cross_bound[:, 0], cross_bound[:, 1]] = -1


        self.ground_dict[key] = (ground_clas, ground_coord)

        assert ground_clas.shape==(1,grid_size[0],grid_size[1])
        assert ground_coord.shape==(4,grid_size[0],grid_size[1])

        return ground_clas, ground_coord



    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self,p_out,n_out):

        #torch.nn.BCELoss()
        # TODO compute classifier's loss

        #TODO double check the dimension here
        p_out_gt = torch.ones_like(p_out)
        n_out_gt = torch.zeros_like(n_out)
        BCELoss = nn.BCELoss()
        # prediction = torch.cat((p_out,n_out))
        # target = torch.cat((p_out_gt, n_out_gt))
        # loss_test = BCELoss(prediction,target)

        loss_p = BCELoss(p_out,p_out_gt)*len(p_out_gt)
        loss_n = BCELoss(n_out, n_out_gt)*len(n_out_gt)

        loss = (loss_p + loss_n)/(len(p_out_gt)+len(n_out_gt))

        sum_count = 0
        return loss,sum_count



    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self,pos_target_coord,pos_out_r, effective_batch=120):
            #torch.nn.SmoothL1Loss()
            # TODO compute regressor's loss

            # TODO double check the dimension here
            SmoothL1Loss = nn.SmoothL1Loss()
            loss = (SmoothL1Loss(pos_out_r,pos_target_coord)) * len(pos_target_coord) / effective_batch
            #TODO not sure what sum_count for
            sum_count = 0
            return loss, sum_count



    # Compute the total loss
    # Input:
    #       clas_out: (bz,1,grid_size[0],grid_size[1])
    #       regr_out: (bz,4,grid_size[0],grid_size[1])
    #       targ_clas:(bz,1,grid_size[0],grid_size[1])
    #       targ_regr:(bz,4,grid_size[0],grid_size[1])
    #       l: lambda constant to weight between the two losses
    #       effective_batch: the number of anchors in the effective batch (M in the handout)
    def compute_loss(self,clas_out,regr_out,targ_clas,targ_regr, l=1, effective_batch=50):
            #############################
            # TODO compute the total loss
            #############################

            ## torch.random.manual_seed(1)

            ## double check sampling

            #flat all 4D tensors to 2D
            clas_out = clas_out.permute(0,2,3,1) # (bz,g_size[0],g_size[1],1)
            regr_out = regr_out.permute(0,2,3,1) # (bz,g_size[0],g_size[1],4)
            targ_clas = targ_clas.permute(0,2,3,1)
            targ_regr = targ_regr.permute(0,2,3,1)

            flat_clas_out = torch.flatten(clas_out, 0,2) #shape: (bz*N,1)
            flat_regr_out = torch.flatten(regr_out,0,2) #shape: (bz*N,4)
            flat_targ_clas = torch.flatten(targ_clas,0,2)
            flat_targ_regr = torch.flatten(targ_regr,0,2)

            #Pick max(M/2,available positive ground truth anchors)
            positive_gt_anchor_idx = torch.where(flat_targ_clas==1)
            negative_gt_anchor_idx = torch.where(flat_targ_clas==0)

            num_pos_anchor = int(min(effective_batch/2,len(positive_gt_anchor_idx[0])))
            num_neg_anchor = effective_batch - num_pos_anchor

            #create random index for pos and negative gt anchor, and pick corresponding # of idx
            pos_rand_idx = torch.randperm(len(positive_gt_anchor_idx[0]))[:num_pos_anchor]
            neg_rand_idx = torch.randperm(len(negative_gt_anchor_idx[0]))[:num_neg_anchor]

            p_classifier_out = flat_clas_out[positive_gt_anchor_idx] #shape: (num_pos_anchor,1)
            # p_classifier_out = p_classifier_out[:num_pos_anchor]
            p_classifier_out = p_classifier_out[pos_rand_idx]
            n_classifier_out = flat_clas_out[negative_gt_anchor_idx] #shape: (effective_batch - num_pos_anchor,1)
            # n_classifier_out = n_classifier_out[: num_neg_anchor]
            n_classifier_out = n_classifier_out[neg_rand_idx]

            p_regressor_gt = flat_targ_regr[positive_gt_anchor_idx[0]] #shape: (num_pos_anchor,4)
            # p_regressor_gt = p_regressor_gt[:num_pos_anchor]
            p_regressor_gt = p_regressor_gt[pos_rand_idx]
            p_regressor_pred = flat_regr_out[positive_gt_anchor_idx[0]] #shape: (num_pos_anchor,4)
            # p_regressor_pred = p_regressor_pred[:num_pos_anchor]
            p_regressor_pred = p_regressor_pred[pos_rand_idx]


            # data used for testing the loss function
            # test_data = torch.load('./HW4_TestCases/Loss/loss_test_' + str(0) + '.pt')
            # sol_p_out = test_data.get('p_out')
            # sol_n_out = test_data.get('n_out')
            # sol_pos_target_coord = test_data.get('pos_target_coord')
            # sol_pos_out_r = test_data.get('pos_out_r')

            l=5

            loss_c,_= self.loss_class(p_classifier_out.to(self.device), n_classifier_out.to(self.device))
            loss_r,_  = self.loss_reg(p_regressor_gt.to(self.device), p_regressor_pred.to(self.device),effective_batch) #effective batch here is N_reg
            # print('my loss r', loss_r)
            # loss_c,_= self.loss_class(sol_p_out, sol_n_out)
            # loss_r, _ = self.loss_reg(sol_pos_target_coord, sol_pos_out_r)  # effective batch here is N_reg
            loss = loss_c+loss_r*l

            return loss, loss_c, loss_r



    # Post process for the outputs for a batch of images
    # Input:
    #       out_c:  (bz,1,grid_size[0],grid_size[1])}
    #       out_r:  (bz,4,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinates of the boxes that the NMS kept)
    def postprocess(self,out_c,out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=5):
       ####################################
       # TODO postprocess a batch of images
       #####################################
        num_img = out_c.shape[0]
        nms_clas_list = []
        nms_prebox_list = []
        for i in range(num_img):
            nms_clas, nms_prebox = self.postprocessImg(out_c[i],out_r[i],IOU_thresh,keep_num_preNMS,keep_num_postNMS)
            nms_clas_list.append(nms_clas)
            nms_prebox_list.append(nms_prebox)

        return nms_clas_list, nms_prebox_list



    # Post process the output for one image
    # Input:
    #      mat_clas: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
    #      mat_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)
    def postprocessImg(self,mat_clas,mat_coord, IOU_thresh,keep_num_preNMS, keep_num_postNMS,image_size=(800,1088)):
            ######################################
            # TODO postprocess a single image
            #####################################

            # flatten  mat_coord & mat_clas  -> (bz*N, 4), (bz*N, 1)
            flat_coord, flat_clas, flat_anchor = output_flattening(mat_coord, mat_clas, self.anchors)

            # decode flat_coord
            # flat_coord = output_decoding(flat_coord.to(self.device), flat_anchor.to(self.device))
            flat_coord = output_decoding(flat_coord, flat_anchor)

            # Crop the cross-boundary proposals !!

            # h, w = image_size
            # cross_bound_w = torch.bitwise_and(flat_coord[:,0] > 0, flat_coord[:,2] < w)
            # cross_bound_h = torch.bitwise_and(flat_coord[:,1] > 0, flat_coord[:,3] < h)
            # cross_bound = torch.bitwise_and(cross_bound_w, cross_bound_h).nonzero()

            #
            # flat_coord = flat_coord[cross_bound.squeeze()]
            # flat_clas = flat_clas[cross_bound.squeeze()]

            # Keep the proposals with the top K objectness scores
            score, ind = torch.topk(flat_clas.flatten(), keep_num_preNMS)
            # score, ind = torch.topk(flat_clas.flatten(), 20)
            topk_clas = flat_clas[ind]   # (N,1)
            topk_coord = flat_coord[ind] # (N,4)

            nms_clas = topk_clas
            nms_prebox = topk_coord

            # Apply NMS on this top K proposals
            nms_clas, nms_prebox = self.NMS(topk_clas, topk_coord, IOU_thresh)

            if len(nms_clas) > keep_num_postNMS:
                nms_clas = nms_clas[:keep_num_postNMS]
                nms_prebox = nms_prebox[:keep_num_postNMS]

            ## add objectiveness threshold = 0.5
            score = nms_clas.flatten()
            nms_clas = nms_clas[score>=0.5]
            nms_prebox = nms_prebox[score>=0.5]

            return nms_clas, nms_prebox



    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self,clas,prebox, thresh):
        ##################################
        # TODO perform NSM
        ##################################

        # get iou score pairwise
        nonSuppresed = torch.ones(len(clas))
        for i in range(len(clas)):
            iou_row = flat_IOU(prebox, prebox[i])
            if len(clas[iou_row > thresh])==0:
                continue
            if (clas[iou_row > thresh] > clas[i]).sum()>0:
                nonSuppresed[i] = 0

        nms_clas = clas[nonSuppresed==1]
        nms_prebox = prebox[nonSuppresed==1]

        return nms_clas,nms_prebox


    #Input:
    # clas_prediction: prediction objectness of anchors, shape: (bz,1 grid_size[0],grid_size[1])
    # clas_gt: ground truth objectness of anchors, shape: (bz,1 grid_size[0],grid_size[1])
    # sigma: threshold for distinguish positiv and negative anchors
    def pointwise_acc(self,clas_prediction,clas_gt,sigma=0.5):

        # set anchor with confidence score lower than threshold as negative
        # clas_prediction = torch.cat(clas_prediction,dim=0)
        # clas_gt = torch.cat(clas_gt,dim=0)


        clas_prediction[clas_prediction <= sigma] = 0
        clas_prediction[clas_prediction > sigma] = 1

        # considering all non-pos and non-neg anchors as negative
        clas_gt[clas_gt<=sigma] = 0
        correct_num = torch.sum(clas_prediction==clas_gt,dim=(1,2,3)) #calculate the number of positive predction per image
        total_num = clas_gt.shape[-2] * clas_gt.shape[-1]

        acc_image = correct_num.float()/total_num

        acc = torch.sum(acc_image) / clas_gt.shape[0] #average the accuracy of all imgs
        return acc

def test_gtlabels(rpn_net):
    path_anchors = '/Users/liuchang/Documents/Fall2020 Academia/CIS680/cis680hws/CIS680_Projects/hw4_MaskRCNN/HW4_TestCases/Anchor_Test/anchors_ratio0.5_scale128.pt'
    gt_anchors = torch.load(path_anchors)
    anchors = rpn_net.create_anchors(aspect_ratio=0.5, scale=128, grid_sizes=(50,68), stride=16)
    print((anchors- gt_anchors).sum())

    # print(anchors)
    path_gt = '/Users/liuchang/Documents/Fall2020 Academia/CIS680/cis680hws/CIS680_Projects/hw4_MaskRCNN/HW4_TestCases/Ground_Truth/ground_truth_index_[786].pt'
    gts = torch.load(path_gt)
    # print(gts['bboxes'])
    gt_cls, gt_coord = gts['ground_clas'], gts['ground_coord']
    cls, coord= rpn_net.create_ground_truth(gts['bboxes'], gts['index'], gts['grid_size'], gts['anchors'].type(torch.float32), gts['image_size'])

    
if __name__=="__main__":

    rpn_net = RPNHead()




