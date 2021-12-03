# -*- coding: UTF-8 -*-
import argparse
import os
import time
from pathlib import Path

import cv2
import torch
import copy
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_corner, scale_coords, xyxy2xywh, increment_path, \
    find_T_matrix
from utils.torch_utils import time_synchronized

lp_transform_dest = [None, np.float32([[0, 0], [470, 0], [470, 110], [0, 110]]),
                     np.float32([[0, 0], [300, 0], [300, 165], [0, 165]])]
lp_crop_size = [None, (470, 110), (300, 165)]


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    return coords


def show_results(img, xywh, conf, landmarks):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for i in range(4):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def crop_affine(img, landmarks, lp_type):
    h, w, c = img.shape
    points = []
    for i in range(4):
        points.append([landmarks[2 * i] * w, landmarks[2 * i + 1] * h])
    points = np.array(points, np.float32)
    M = cv2.getPerspectiveTransform(points, lp_transform_dest[lp_type])
    dst = cv2.warpPerspective(img, M, lp_crop_size[lp_type])
    return dst

def detect_one(model, image_path, device, img_size, vis=True):
    # Load model
    conf_thres = 0.7
    iou_thres = 0.3

    print(image_path)
    orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    img1 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    img = letterbox(img0, new_shape=imgsz)[0]
    # img = img0
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_corner(pred, conf_thres, iou_thres)

    # print('img.shape: ', img.shape)
    # print('orgimg.shape: ', orgimg.shape)

    import math

    # Process detections
    crop_images = []
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()
            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:13].view(1, 8) / gn_lks).view(-1).tolist()
                # a = max(math.dist(landmarks[0:2], landmarks[2:4]), math.dist(landmarks[4:6], landmarks[6:8]))*w0
                # b = min(math.dist(landmarks[0:2], landmarks[6:8]), math.dist(landmarks[4:6], landmarks[2:4]))*h0
                # print(a/b, w0, h0)
                # lp_type = 1 if a / b > 2.3 else 2
                crop_images.append(crop_affine(img1, landmarks, 2))
                if vis:
                    orgimg = show_results(orgimg, xywh, conf, landmarks)

    t2 = time.time()
    print('runtime:', t2 - t0, t2 - t1, '\n')

    # cv2.imwrite('result.jpg', orgimg)
    return orgimg, crop_images


def detect():
    img_ext = ['jpg', 'png']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    if os.path.isfile(opt.image):
        result = detect_one(model, opt.image, device, opt.img_size, vis=False)
        save_path = os.path.join(opt.save_dir, os.path.basename(opt.image))
        cv2.imwrite(save_path, result)
    elif os.path.isdir(opt.image):
        for image_name in os.listdir(opt.image):
            if image_name[-3:] not in img_ext:
                continue
            image_path = os.path.join(opt.image, image_name)
            result, crop_images = detect_one(model, image_path, device, opt.img_size, vis=True)
            save_path = os.path.join(opt.save_dir, image_name)
            cv2.imwrite(save_path, result)
            for i, crop_img in enumerate(crop_images):
                cv2.imwrite(save_path[:-3]+'_{}.jpg'.format(i), crop_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt',
                        help='model.pt path(s)')
    parser.add_argument('--image', type=str, default='data/images/test.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--project', default='runs/inference', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    if not os.path.isdir(opt.save_dir):
        os.makedirs(opt.save_dir)
    print('Saving result to', opt.save_dir)
    detect()
