import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial
import pdb
import copy

from BoxHead import *
from pretrained_models import *
# from pretrained_models import *
import os
import time
import torch.optim as optim
from torchvision.models.detection.image_list import ImageList

def load_data(batch_size=2):
    '''
    Load data from file, split, then create data loader
    :param: batch_size
    :return: train_loader, test_loader
    '''

    #TODO change the path
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size

    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # push the randomized training data into the dataloader
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    return train_loader, test_loader

def box_train(box_head, train_loader,optimizer,epoch,backbone,rpn,keep_topK):
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_loss = 0
    epoch_clas_loss = 0
    epoch_regr_loss = 0
    running_loss =0
    running_clas_loss =0
    running_regr_loss =0

    # TODO double check following two values, just placehoder for now
    l = 1
    effective_batch = 32  # used in TA's test case

    for i,data in enumerate(train_loader):

        optimizer.zero_grad()
        # images, labels, mask, boxes, indexes = [data[key] for key in data.keys()]
        # images = images.to(device)
        images = data['images'].to(device)
        indexes = data['index']
        boxes = data['bbox']
        labels = data['labels']
        mask = data['masks']

        # Take the features from the backbone
        backout = backbone(images)
        # The RPN implementation takes as first argument the following image list
        im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
        # Then we pass the image list and the backbone output through the rpn
        rpnout = rpn(im_lis, backout)
        #The final output is
        # A list of proposal tensors: list:len(bz){(keep_topK,4)}
        proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
        # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
        fpn_feat_list= list(backout.values())
        # fpn_feat_list =  [item.to('cpu') for item in fpn_feat_list]

        gt_labels, gt_regressor_target = box_head.create_ground_truth(proposals,labels,boxes)

        #TOdo check this line

        # proposals_roi = copy.deepcopy(proposals)
        roi_align_result = box_head.MultiScaleRoiAlign(fpn_feat_list,proposals) #This is the input to Box head
        clas_out, regr_out = box_head.forward(roi_align_result.to(device))

        # for j in range(1):
        #     label = labels[j]
        #     print('num of objects', len(labels[j]))
        #     bbox = boxes[j]
        #     img_squeeze = transforms.functional.normalize(images[j,:,:,:].to('cpu'),
        #                                                   [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        #                                                   [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
        #     fig,ax=plt.subplots(1,1)
        #     ax.imshow(img_squeeze.permute(1,2,0))
        #     for box in proposals[j]:
        #         box=box.view(-1)
        #         rect=patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color='b')
        #         ax.add_patch(rect)
        #     plt.show()
        loss, loss_c, loss_r = box_head.compute_loss(clas_out,regr_out,gt_labels,gt_regressor_target,l,effective_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss#.item()
        epoch_clas_loss += loss_c#.item()
        epoch_regr_loss += loss_r
        running_loss += loss#.item()
        running_clas_loss += loss_c#.item()
        running_regr_loss += loss_r

        #print results every log_iter batch:
        log_iter = 100
        if i % log_iter == (log_iter-1):  # print every 100 mini-batches
            print('[%d, %5d] total_loss: %.5f clas_loss: %.5f  regr_loss: %.5f' %
                  (epoch + 1, i + 1,
                  running_loss / log_iter,
                  running_clas_loss / log_iter,
                  running_regr_loss / log_iter))

            running_loss = 0
            running_clas_loss = 0
            running_regr_loss = 0
            print("--- %s minutes ---" % ((time.time() - start_time)/60))
            start_time = time.time()


        #delete variables after usage to free GPU ram, double check if these variables are needed for future!!!!!!!
        # del loss ,loss_c , loss_r
        # del images, labels, mask, boxes, indexes
        # del clas_out, regr_out
        # del gt_labels, gt_regressor_target
        # torch.cuda.empty_cache()


    epoch_loss /= i
    epoch_clas_loss /= i
    epoch_regr_loss /= i

    return epoch_loss, epoch_clas_loss, epoch_regr_loss

# def rpn_validation(rpn_head, test_loader,optimizer,epoch):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     epoch_loss = 0
#     epoch_clas_loss = 0
#     epoch_regr_loss = 0
#
#     # TODO double check following two values, just placehoder for now
#     l = 1
#     effective_batch = 120  # suggestd 150 for 4 images
#
#     for i,data in enumerate(test_loader):
#         imgs, label_list, mask_list, bbox_list, index_list = [data[key] for key in data.keys()]
#         with torch.no_grad():
#             clas_out, regr_out = rpn_head.forward(imgs.to(device))
#         ##Todo check this line
#             target_clas, target_regr = rpn_head.create_batch_truth(bbox_list,index_list,imgs.shape[-2:])
#             loss, loss_c, loss_r = rpn_head.compute_loss(clas_out,regr_out,target_clas,target_regr,l,effective_batch)
#         epoch_loss += loss.item()
#         epoch_clas_loss += loss_c.item()
#         epoch_regr_loss += loss_r.item()
#
#         #delete variables after usage to free GPU ram, double check if these variables are needed for future!!!!!!!
#         del loss ,loss_c , loss_r
#         del imgs, label_list, mask_list, bbox_list, index_list
#         del clas_out, regr_out
#         del target_clas, target_regr
#         torch.cuda.empty_cache()
#
#     epoch_loss /= i
#     epoch_clas_loss /= i
#     epoch_regr_loss /= i
#
#     return epoch_loss, epoch_clas_loss, epoch_regr_loss


def main():

    pretrained_path = 'checkpoint680.pth'
    backbone, rpn = pretrained_models_680(pretrained_path)
    # we will need the ImageList from torchvision
    from torchvision.models.detection.image_list import ImageList

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 4 #suggested 1-4
    print(device)
    ## load data
    train_loader, test_loader = load_data(batch_size)
    print('data loaded')

    # Here we keep the top 20, but during training you should keep around 200 boxes from the 1000 proposals
    keep_topK= 200

    box_head = BoxHead().to(device)



    ## initialize optimizer
    learning_rate = 1e-3  # for batch size 2

    #TA suggest Adam with 0.001 lr with 40 epoch
    optimizer = optim.Adam(box_head.parameters(),lr=learning_rate)

    num_epochs = 50
    print('.... start training ....')

    #created lists to store losses during the training process
    total_loss_list = []
    classifier_loss_list = []
    regressor_loss_list = []

    #lists store values for validation
    val_total_loss_list = []
    val_classifier_loss_list = []
    val_regresor_loss_list = []

    for epoch in range(num_epochs):


        total_loss, classifier_loss, regressor_loss = box_train(box_head, test_loader, optimizer,epoch,backbone,rpn,keep_topK)
        total_loss_list.append(total_loss)
        classifier_loss_list.append(classifier_loss)
        regressor_loss_list.append(regressor_loss)

    #     val_total_loss, val_classifier_loss, val_regresor_loss = rpn_validation(rpn_head, test_loader, optimizer,epoch)
    #     val_total_loss_list.append(val_total_loss)
    #     val_classifier_loss_list.append(val_classifier_loss)
    #     val_regresor_loss_list.append(val_regresor_loss)
    #
    #     train_loss = [total_loss_list,classifier_loss_list,regressor_loss_list]
    #     test_loss = [val_total_loss_list,val_classifier_loss_list,val_regresor_loss_list]
    #
    #     print('**** epoch loss ****', total_loss, classifier_loss, regressor_loss,'**** validation ****', \
    #           val_total_loss, val_classifier_loss, val_regresor_loss)
    #     path = os.path.join('./rpn_results/','rpn_epoch_1_' + str(epoch))
    #     torch.save({
    #         'epoch': epoch ,
    #         'model_state_dict': rpn_head.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'train_loss': train_loss,
    #         'test_loss': test_loss
    #     }, path)
    # print('finished training')
    # pass

if __name__ == '__main__':

    main()