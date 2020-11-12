import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
from rpn import *
from BoxHead import *
from utils import *
import matplotlib.patches as patches


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        #############################################
        # TODO Initialize  Dataset
        #############################################

        self.path = path
        imgs_path, masks_path, labels_path, bboxes_path = self.path
        self.imgs_data = h5py.File(imgs_path, 'r').get('data')
        input_masks_data = h5py.File(masks_path, 'r').get('data')
        self.masks_data = []
        self.labels_data = np.load(labels_path, allow_pickle=True, encoding="latin1")
        self.bboxes_data = np.load(bboxes_path, allow_pickle=True)
        idx = 0
        # linking the mask and image
        for i in range(len(self.imgs_data)):
            num_mask = len(self.labels_data[i])
            self.masks_data.append(input_masks_data[idx:num_mask + idx, :, :])
            idx += num_mask

    # In this function for given index we rescale the image and the corresponding  masks, boxes
    # and we return them as output
    # output:
    # transed_img
    # label
    # transed_mask
    # transed_bbox
    # index
    def __getitem__(self, index):
        ################################
        # TODO return transformed images,labels,masks,boxes,index
        ################################

        # load datapoint wrt given index
        img = torch.tensor(self.imgs_data[index].astype(np.float32), dtype=torch.float)
        bbox = torch.tensor(self.bboxes_data[index], dtype=torch.float)
        label = torch.tensor(self.labels_data[index], dtype=torch.float)
        mask = torch.tensor(self.masks_data[index].astype(np.float32), dtype=torch.float)

        # normalize, rescale the img and mask
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(img, mask, bbox)

        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]

        return transed_img, label, transed_mask, transed_bbox, index

    # This function preprocess the given image, mask, box by rescaling them appropriately
    # output:
    #        img: (3,800,1088)
    #        mask: (n_box,800,1088)
    #        box: (n_box,4)
    def pre_process_batch(self, img, mask, bbox):
        #######################################
        # TODO apply the correct transformation to the images,masks,boxes
        ######################################

        # normalize the pixel value to [0,1]
        img /= 255.0
        # img = torch.tensor((img / 255.0), dtype=torch.float)

        # rescale the image to 800x1066 (we want to keep the aspect ratio the same)
        img = F.interpolate(img, size=1066)
        img = img.permute(0, 2, 1)
        img = F.interpolate(img, size=800)
        img = img.permute(0, 2, 1)

        # normalize each channel
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=False)
        img = normalize(img)
        # add zero padding to get an image of size 800x1088
        img = F.pad(img, (11, 11))

        # rescale mask
        mask = F.interpolate(mask, size=1066)
        mask = mask.permute(0, 2, 1)
        mask = F.interpolate(mask, size=800)
        mask = mask.permute(0, 2, 1)
        mask = F.pad(mask, (11, 11))

        # scaling and shifting box
        bbox = bbox * 8 / 3
        bbox[:, 0] += 11
        bbox[:, 2] += 11

        # check flag
        assert img.squeeze(0).shape == (3, 800, 1088)
        # print(bbox, mask)
        assert bbox.shape[0] == mask.shape[0]

        return img.squeeze(0), mask, bbox

    def __len__(self):
        return len(self.imgs_data)


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):

        out_batch = {'images':[], 'labels':[], 'masks':[],'bbox':[], 'index':[]}

        for transed_img, label, transed_mask, transed_bbox, ind in batch:
            out_batch['images'].append(transed_img)
            out_batch['labels'].append(label)
            out_batch['masks'].append(transed_mask)
            out_batch['bbox'].append(transed_bbox)
            out_batch['index'].append(ind)

        out_batch['images'] = torch.stack(out_batch['images'],dim=0)
        return out_batch

    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)

def plot_datapoint(images, bboxes, masks, labels, inds):
    '''
    given a batch of datapoints -> plot single datapoint one by one <img, bounding box, colored mask> for submission 1)
    :param: img, list of torch: len(bz); torch: (3, 800, 1080)
    :param: bboxs: list of torch: len(bz); torch: (nobj, 4)
    :param: masks: list of torch: len(bz); torch: (nobj, 800, 1080)
    :param: labels: list of torch: len(bz); torch: (nobj, )
    :param: inds: list of int: len(bz)
    '''
    mask_color_list = ["jet", "ocean", "Spectral"] # vehicle/blue, human/green, animal/red
    mask_color_list = ["jet", "Spectral", "ocean"] # vehicle/blue, human/red, animal/green
    edge_color_list = ["b", "r", "g"]
    label_names = ['vehicle', 'human', 'animal']

    for i in range(len(images)):

        # turn img from (3,800,1088)->(800, 1088, 3)
        image = images[i]
        im = np.zeros((800, 1088, 3))
        im[:, :, 0] = torch.clamp(image[0, :, :],0,1)
        im[:, :, 1] = torch.clamp(image[1, :, :],0,1)
        im[:, :, 2] = torch.clamp(image[2, :, :],0,1)
        ind = inds[i]
        fig, ax = plt.subplots(1)
        ### Display the image ###
        ax.imshow(np.squeeze(im))

        for j in range(len(bboxes[i])):
            cls = labels[i][j]
            box = bboxes[i][j]
            mask = masks[i][j]
            label_ind = int(labels[i][j] - 1)
            # Create a Rectangle patch
            x, y = box[0], box[1]
            w = (box[2] - box[0])
            h = (box[3] - box[1])

            rect = patches.Rectangle((x, y), w, h,
                                     linewidth=1, edgecolor=edge_color_list[label_ind], facecolor='none')

            # add label to bbox
            plt.annotate(label_names[label_ind], xy=(x,y), xycoords='data',color='y')

            # Add the patch to the Axes
            ax.add_patch(rect)

            # plot mask
            mask = np.ma.masked_where(mask == 0, mask)
            plt.imshow(mask, cmap=mask_color_list[int(cls) - 1], alpha=0.6)

        plt.savefig("./testfig/visualtrainset"+str(ind)+".png")
        plt.show()




if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    print('full_size',full_size)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    rpn_net = RPNHead()
    box_head = BoxHead()
    # push the randomized training data into the dataloader
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    # additional functions
    function = 'plot gt'
    if function == 'plot datapoint':
        for i, batch in enumerate(train_loader, 0):
            images = batch['images']
            indexes = batch['index']
            boxes = batch['bbox']
            masks = batch['masks']
            labels = batch['labels']
            plot_datapoint(images, boxes, masks,labels, indexes)
            if i == 3:
                exit()
    elif function == 'histogram':
        ratio = []
        wh = []
        for i, batch in enumerate(train_loader, 0):
            bz_boxes = batch['bbox']
            for boxes in bz_boxes:
                for box in boxes:
                    w = abs(box[2]- box[0])
                    h = abs(box[3] - box[1])
                    ratio.append((w/h).item())
                    wh.append(((w*h)**0.5).item())

        plt.hist(ratio, bins=np.linspace(0.2, 2, 20))
        plt.ylabel('number')
        plt.xlabel('ratio')
        plt.title('ratio histagram')
        plt.savefig("./testfig/ratio_histagram" + ".png")
        plt.show()

        plt.hist(wh, bins=np.linspace(100, 800, 20))

        plt.ylabel('number')
        plt.xlabel('wh')
        plt.title('wh histagram')
        plt.savefig("./testfig/wh_histagram" + ".png")
        plt.show()

        exit()


    for i, batch in enumerate(train_loader, 0):
        #      bboxes_list: list:len(bz){(n_obj,4)}
        #      indexes:      list:len(bz)
        #      image_shape:  tuple:len(2)
        images = batch['images'][0]
        indexes = batch['index']
        boxes = batch['bbox']
        labels = batch['labels']

        if function == 'plot gt':
            gt, ground_coord = rpn_net.create_batch_truth(boxes, indexes, images[0].shape[-2:])
            proposals = rpn_net.forward(images)
            gt_labels, gt_regressor_target = box_head.create_ground_truth(proposals,labels,boxes)
            decoded_gt_box = output_decoding(flatten_coord, flatten_anchors)

            obj_idx = torch.where(gt_labels != 0)
            #visualizing the ground truth box
            color_list = ["jet", "ocean", "Spectral"]
            images = transforms.functional.normalize(images,
                                                     [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                     [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(images.permute(1, 2, 0))
            #visualizing the ground truth box in one img
            for i in range(len(labels)):
                class_label = labels[i]
                box = boxes[i]
                gt_box = paths.Rectangle(box[0],box[1],box[2]-box[0],box[3]-box[1], fill=False, color = color_list[class_label-1])
                ax.add_patch(gt_box)
                plt.show()
            #visualizing proposals for each class
            for i in range(3):
                fig, ax = plt.subplots(1,1)
                ax.imshow(images.permute(1,2,0))
                class_idx = torch.where(gt_labels==i+1)
                class_box = decoded_gt_box[class_idx]
                for box in class_box:
                    proposal_box = paths.Rectangle(box[0],box[1],box[2],box[3], fill = False, color = color_list[i])
                    ax.add_patch(proposal_box)
                plt.show()
            # for idx in obj_idx:
            #     obj_proposal = proposals[idx]
            #     obj_gt_label = gt_labels[idx]
            #     obj_gt_box = decoded_gt_box[idx]
            #
            #     gt_rect = paths.Rectangle(obj_gt_box[0],obj_gt_box[1],obj_gt_box[2],obj_gt_box[3], \
            #                               fill = False, color = 'r')
            #     proposal_rect = paths.Rectangle(obj_proposal[0],obj_proposal[1],obj_proposal[2]-obj_proposal[0], \
            #                                     obj_proposal[3]-obj_proposal[1], \
            #                                     fill = False, color = color_list[obj_gt_box])
            #     ax.add_patch(gt_rect)
            #     ax.add_patch(proposal_rect)
            #
            #
            # # plt.savefig("./testfig/groundtruth_" + str(i) + ".png")
            # plt.show()

            if (i > 8):
                break


            # Flatten the ground truth and the anchors
            flatten_coord, flatten_gt, flatten_anchors = output_flattening(ground_coord, gt, rpn_net.get_anchors())

            # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
            decoded_coord = output_decoding(flatten_coord, flatten_anchors)

            # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
            images = transforms.functional.normalize(images,
                                                     [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                     [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(images.permute(1, 2, 0))

            find_cor = (flatten_gt == 1).nonzero()
            find_neg = (flatten_gt == -1).nonzero()

            for elem in find_cor:
                coord = decoded_coord[elem, :].view(-1)
                anchor = flatten_anchors[elem, :].view(-1)

                col = 'r'
                rect = patches.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1], fill=False,
                                         color=col)
                ax.add_patch(rect)
                rect = patches.Rectangle((anchor[0] - anchor[2] / 2, anchor[1] - anchor[3] / 2), anchor[2], anchor[3],
                                         fill=False, color='b')
                ax.add_patch(rect)

            plt.savefig("./testfig/groundtruth_" + str(i) + ".png")
            plt.show()

            if (i > 8):
                break
        elif function == 'plot postprocess':
            images = batch['images']
            clas_out, regr_out = rpn_net.forward(images)

            nms_clas_list, nms_prebox_list = rpn_net.postprocess(clas_out, regr_out)

            flatten_gt = nms_clas_list[0]
            decoded_coord = nms_prebox_list[0]
            print(decoded_coord)

            # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
            images = transforms.functional.normalize(images[0],
                                                     [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                     [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(images.permute(1, 2, 0))


            for elem in range(len(flatten_gt)):
                coord = decoded_coord[elem, :].view(-1)
                # anchor = flatten_anchors[elem, :].view(-1)

                col = 'b'
                rect = patches.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1], fill=False,
                                         color=col)
                ax.add_patch(rect)
                # rect = patches.Rectangle((anchor[0] - anchor[2] / 2, anchor[1] - anchor[3] / 2), anchor[2], anchor[3],
                #                          fill=False, color='b')
                # ax.add_patch(rect)

            # plt.savefig("./testfig/postprocess_" + str(i) + ".png")
            plt.show()

            if (i > 3):
                break


