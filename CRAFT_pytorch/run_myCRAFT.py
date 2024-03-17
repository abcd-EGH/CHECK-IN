"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
from IPython.display import display

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
import base64
from io import BytesIO

import cv2
import numpy as np
try:
    import CRAFT_pytorch.craft_utils as craft_utils
    import CRAFT_pytorch.imgproc as imgproc

    from CRAFT_pytorch.craft import CRAFT
except:
    import craft_utils as craft_utils
    import imgproc as imgproc

    from craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def process_image_pil(image_pil):
    # PIL 이미지를 numpy 배열로 변환
    img = np.array(image_pil)

    # RGB 순서 확인 및 조정 (필요한 경우에만 적용)
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4: img = img[:, :, :3]

    return img


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):

    # 이미지의 원래 크기 저장
    size = image.shape[:2]  # (height, width)

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    return boxes, polys, size

def convert_polys_to_bytes(polys):
    polys_bytes = []
    for poly in polys:
        if isinstance(poly, np.ndarray):
            # NumPy 배열을 바이트로 변환
            poly_bytes = poly.tobytes()
            # 메타데이터 (형태와 데이터 타입) 저장
            meta_data = {'shape': poly.shape, 'dtype': str(poly.dtype)}
            # 바이트 데이터와 메타데이터를 함께 저장
            polys_bytes.append({'bytes': poly_bytes, 'meta': meta_data})
        else:
            # NumPy 배열이 아닌 경우의 처리
            pass
    return polys_bytes

def return_path_poly_size(base_path=None, image_path=None, image_PIL=None, cuda=True):
    # load net
    net = CRAFT(base_path = base_path)     # initialize

    # print('Loading weights from checkpoint (' + base_path + ')')
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(base_path)))
    else:
        net.load_state_dict(copyStateDict(torch.load(base_path, map_location='cpu')))

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None

    if image_path:
        image = imgproc.loadImage(image_path)
        bboxes, polys, size = test_net(net, image, 0.7, 0.4, 0.4, cuda, False, refine_net)
    elif image_PIL:
        image = process_image_pil(image_PIL)
        bboxes, polys, size = test_net(net, image, 0.7, 0.4, 0.4, cuda, False, refine_net)
    else:
        raise "No imput image"

    return image_path, polys, size

# image_path, polys, _ = return_path_poly_size(base_path='./CRAFT_pytorch/craft_mlt_25k.pth', image_path='./CRAFT_pytorch/test/nontarget.jpg', cuda=torch.cuda.is_available())

# polys로 bounding boxes 그리기
def draw_bounding_boxes(image, bounding_boxes):
    # 이미지 복사
    image_copy = image.copy()

    # 이미지에 바운딩 박스를 그리는 함수
    for box in bounding_boxes:
        # 각 박스에 대해 사각형 그리기
        cv2.polylines(image_copy, [np.array(box, np.int32).reshape((-1, 1, 2))], isClosed=True, color=(255, 0, 0), thickness=2)
    
    return image_copy

# 바운딩 박스를 그린 후 base64 인코딩
def encode_image_to_base64(image_path, bounding_boxes):
    # 이미지 불러오기
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 바운딩 박스 그리기 (원본 이미지는 수정되지 않음)
    image = draw_bounding_boxes(image, bounding_boxes)

    # 이미지를 PIL Image 객체로 변환
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 이미지를 byte로 변환 후 Base64 인코딩
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return encoded_string

# encoded_string = encode_image_to_base64(image_path, polys)


