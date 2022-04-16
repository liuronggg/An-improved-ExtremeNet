import os
import cv2
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from config import system_configs
from utils import crop_image, normalize_
from utils.visualize import feature
from external.nms import soft_nms_with_points as soft_nms

from torch import nn
import torch.nn.functional as F

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU())
    return blk

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

net = nn.Sequential(
    nin_block(3, 96, kernel_size=3, stride=1, padding=0),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nin_block(96, 256, kernel_size=3, stride=1, padding=0),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout(0.5),
    nin_block(256, 2, kernel_size=3, stride=1, padding=0),
    GlobalAvgPool2d(), 
    FlattenLayer())


def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs /= ratios[:, 1][:, None, None]
    ys /= ratios[:, 0][:, None, None]
    xs -= borders[:, 2][:, None, None]
    ys -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)


def _rescale_ex_pts(detections, ratios, borders, sizes):
    xs, ys = detections[..., 5:13:2], detections[..., 6:13:2]
    xs /= ratios[:, 1][:, None, None]
    ys /= ratios[:, 0][:, None, None]
    xs -= borders[:, 2][:, None, None]
    ys -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)


def save_image(data, fn):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data)
    plt.savefig(fn, dpi=height)
    plt.close()


def _box_inside(box2, box1):
    inside = (box2[0] >= box1[0] and box2[1] >= box1[1] and \
              box2[2] <= box1[2] and box2[3] <= box1[3])
    return inside


def kp_decode(nnet, images, K, kernel=3, aggr_weight=0.1,
              scores_thresh=0.1, center_thresh=0.1, debug=False):
    detections = nnet.test(
        [images], kernel=kernel, aggr_weight=aggr_weight,
        scores_thresh=scores_thresh, center_thresh=center_thresh, debug=debug)
    detections = detections.data.cpu().numpy()
    return detections


def kp_detection(db, nnet, result_dir, debug=False, decode_func=kp_decode):
    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    if db.split != "trainval":
        db_inds = db.db_inds[:100] if debug else db.db_inds
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds[:5000]
    num_images = db_inds.size

    K = db.configs["top_k"]
    aggr_weight = db.configs["aggr_weight"]
    scores_thresh = db.configs["scores_thresh"]
    center_thresh = db.configs["center_thresh"]
    suppres_ghost = db.configs["suppres_ghost"]
    nms_kernel = db.configs["nms_kernel"]

    scales = db.configs["test_scales"]
    categories = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]

    #cluster_radius = db.configs["cluster_radius"]
    cluster_radius = 149.9

    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1,
        "exp_soft_nms": 2
    }[db.configs["nms_algorithm"]]

    top_bboxes = {}


    for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
        db_ind = db_inds[ind]

        image_id = db.image_ids(db_ind)
        image_file = db.image_file(db_ind)
        image = cv2.imread(image_file)

        height, width = image.shape[0:2]

        detections = []

        for scale in scales:
            new_height = int(height * scale)
            new_width = int(width * scale)
            new_center = np.array([new_height // 2, new_width // 2])

            inp_height = new_height | 127
            inp_width = new_width | 127

            images = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
            ratios = np.zeros((1, 2), dtype=np.float32)
            borders = np.zeros((1, 4), dtype=np.float32)
            sizes = np.zeros((1, 2), dtype=np.float32)

            out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
            height_ratio = out_height / inp_height
            width_ratio = out_width / inp_width

            resized_image = cv2.resize(image, (new_width, new_height))
            resized_image, border, offset = crop_image(
                resized_image, new_center, [inp_height, inp_width])

            resized_image = resized_image / 255.
            normalize_(resized_image, db.mean, db.std)

            images[0] = resized_image.transpose((2, 0, 1))
            borders[0] = border
            sizes[0] = [int(height * scale), int(width * scale)]
            ratios[0] = [height_ratio, width_ratio]

            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            images = torch.from_numpy(images)
            dets = decode_func(
                nnet, images, K, aggr_weight=aggr_weight,
                scores_thresh=scores_thresh, center_thresh=center_thresh,
                kernel=nms_kernel, debug=debug)
            dets = dets.reshape(2, -1, 14)
            dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
            dets[1, :, [5, 7, 9, 11]] = out_width - dets[1, :, [5, 7, 9, 11]]
            dets[1, :, [7, 8, 11, 12]] = dets[1, :, [11, 12, 7, 8]].copy()
            dets = dets.reshape(1, -1, 14)

            _rescale_dets(dets, ratios, borders, sizes)
            _rescale_ex_pts(dets, ratios, borders, sizes)
            dets[:, :, 0:4] /= scale
            dets[:, :, 5:13] /= scale
            detections.append(dets)

        detections = np.concatenate(detections, axis=1)

        classes = detections[..., -1]
        classes = classes[0]
        detections = detections[0]

        keep_inds = (detections[:, 4] > 0)
        detections = detections[keep_inds]
        classes = classes[keep_inds]

        top_bboxes[image_id] = {}

        for j in range(categories):
            keep_inds = (classes == j)
            top_bboxes[image_id][j + 1] = detections[keep_inds].astype(np.float32)


        for j in range(1, categories + 1):
            keep=[]
            i=0
            for bbox in top_bboxes[image_id][j]:
                b = bbox[0:4].astype(np.int32)
                x1 = b[0]
                y1 = b[1]
                x2 = b[2]
                y2 = b[3]
                width = abs(x1 - x2) + 1
                height = abs(y1 - y2) + 1
                if width > height:
                    len_max = width
                    len_min = height
                else:
                    len_max = height
                    len_min = width
                ratio = len_max / len_min
                if len_max < 10:
                    keep.append(i)
                elif 10 <= len_max and len_max <= 13:
                    if ratio >= 1.5:
                        keep.append(i)
                elif 14 <= len_max and len_max <= 17:
                    if ratio >= 2:
                        keep.append(i)
                elif 18 <= len_max and len_max <= 27:
                    if ratio >= 2.5:
                        keep.append(i)
                elif 28 <= len_max and len_max <= 35:
                    if ratio >= 3:
                        keep.append(i)
                elif 36 <= len_max and len_max <= 46:
                    if ratio >= 3.5:
                        keep.append(i)
                elif len_max > 46:
                    keep.append(i)
                i = i+1
            top_bboxes[image_id][j] = np.delete(top_bboxes[image_id][j],keep,axis=0)
        

        for j in range(categories):
            soft_nms(top_bboxes[image_id][j + 1],
                     Nt=nms_threshold, method=nms_algorithm)
        

        previous_n_bboxes = top_bboxes[image_id][1]
        image = cv2.imread(image_file)
        net.load_state_dict(torch.load(r'nin/nin.pth'))
        margin = 5
        for p_bbox in top_bboxes[image_id][2]:
            p_b = p_bbox[0:4].astype(np.int32)
            p_x1 = p_b[0]
            p_y1 = p_b[1]
            p_x2 = p_b[2]
            p_y2 = p_b[3]
            p_area = (p_x2-p_x1+1)*(p_y2-p_y1+1)
            for n_bbox in previous_n_bboxes:
                n_b = n_bbox[0:4].astype(np.int32)
                n_x1 = n_b[0]
                n_y1 = n_b[1]
                n_x2 = n_b[2]
                n_y2 = n_b[3]
                n_area = (n_x2-n_x1+1)*(n_y2-n_y1+1)
                i_x1 = max(n_bbox[0], p_bbox[0])
                i_y1 = max(n_bbox[1], p_bbox[1])
                i_x2 = min(n_bbox[2], p_bbox[2])
                i_y2 = min(n_bbox[3], p_bbox[3])
                if i_x2 >= i_x1 and i_y2 >= i_y1:
                    i_area = (i_x2-i_x1+1)*(i_y2-i_y1+1)
                    if i_area / (n_area + p_area - i_area) > 0.3:
                        p_x1 = p_x1 - margin
                        p_y1 = p_y1 - margin
                        p_x2 = p_x2 + margin
                        p_y2 = p_y2 + margin
                        p_width = p_x2 - p_x1 + 1
                        p_height = p_y2 - p_y1 + 1
                        p_little_img = np.empty([p_height, p_width, 3], dtype = np.uint8)
                        for i in range(p_x1, p_x2+1):
                            for j in range(p_y1, p_y2+1):
                                if i < 0 or j < 0 or i > 127 or j > 127:
                                    p_little_img[j-p_y1, i-p_x1, :] = 255
                                else:
                                    p_little_img[j-p_y1, i-p_x1, :] = image[j, i, :]
                        p_little_img = cv2.cvtColor(p_little_img, cv2.COLOR_BGR2RGB)
                        p_little_img = cv2.resize(p_little_img, (34, 34))
                        p_little_img = p_little_img.astype(np.float32) / 255.
                        p_little_img = p_little_img.transpose((2, 0, 1))
                        p_little_img = torch.Tensor(p_little_img)
                        p_little_img = p_little_img.view((1, p_little_img.shape[0], p_little_img.shape[1], p_little_img.shape[2]))
                        p_cate = net(p_little_img)
                        p_con = p_cate[0][1] / (p_cate[0][0] + p_cate[0][1])
                        if p_con < 0.5:
                            p_bbox[4] = p_bbox[4]*p_con/0.5
                        
                        break


        scores = np.hstack([top_bboxes[image_id][j][:, 4]for j in range(1, categories + 1)])
        if len(scores) > max_per_image:
            kth = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, 4] >= thresh)
                top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]


        '''
        for j in range(1, categories + 1):
            keep=[]
            i=0
            for bbox in top_bboxes[image_id][j]:
                sc = bbox[4]
                ex = bbox[5:13].astype(np.int32).reshape(4, 2)
                feature_val = feature(ex)
                if feature_val > cluster_radius:
                    keep.append(i)
                i = i+1
            
            top_bboxes[image_id][j] = np.delete(top_bboxes[image_id][j],keep,axis=0)
        '''


        if suppres_ghost:
            for j in range(1, categories + 1):
                n = len(top_bboxes[image_id][j])
                for k in range(n):
                    inside_score = 0
                    if top_bboxes[image_id][j][k, 4] > 0.2:
                        for t in range(n):
                            if _box_inside(top_bboxes[image_id][j][t],
                                           top_bboxes[image_id][j][k]):
                                inside_score += top_bboxes[image_id][j][t, 4]
                        if inside_score > top_bboxes[image_id][j][k, 4] * 3:
                            top_bboxes[image_id][j][k, 4] /= 2

        if debug:

            image_file = db.image_file(db_ind)
            image = cv2.imread(image_file)

            bboxes = {}
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, 4] > 0.3)
                cat_name = db.class_name(j)
                cat_size = cv2.getTextSize(
                    cat_name + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                color = np.random.random((3,)) * 0.6 + 0.4
                color = color * 255
                color = color.astype(np.int32).tolist()
                for bbox in top_bboxes[image_id][j][keep_inds]:

                    sc = bbox[4]
                    bbox = bbox[0:4].astype(np.int32)
                    txt = '{}{:.0f}'.format(cat_name, sc * 10)
                    if bbox[1] - cat_size[1] - 2 < 0:
                        cv2.rectangle(image,
                                      (bbox[0], bbox[1] + 2),
                                      (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                                      color, -1
                                      )
                        cv2.putText(image, txt,
                                    (bbox[0], bbox[1] + cat_size[1] + 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                                    thickness=1, lineType=cv2.LINE_AA
                                    )
                    else:
                        cv2.rectangle(image,
                                      (bbox[0], bbox[1] - cat_size[1] - 2),
                                      (bbox[0] + cat_size[0], bbox[1] - 2),
                                      color, -1
                                      )
                        cv2.putText(image, txt,
                                    (bbox[0], bbox[1] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                                    thickness=1, lineType=cv2.LINE_AA
                                    )
                    cv2.rectangle(image,
                                  (bbox[0], bbox[1]),
                                  (bbox[2], bbox[3]),
                                  color, 2
                                  )
            debug_file = os.path.join(debug_dir, "{}.jpg".format(db_ind))
            cv2.imwrite(debug_file, image)
            cv2.imshow('out', image)
            cv2.waitKey()

    result_json = os.path.join(result_dir, "results.json")
    detections = db.convert_to_coco(top_bboxes)
    with open(result_json, "w") as f:
        json.dump(detections, f)

    cls_ids = list(range(1, categories + 1))
    image_ids = [db.image_ids(ind) for ind in db_inds]
    db.evaluate(result_json, cls_ids, image_ids)
    return 0


def testing(db, nnet, result_dir, debug=False):


    return globals()[system_configs.sampling_function](
        db, nnet, result_dir, debug=debug)