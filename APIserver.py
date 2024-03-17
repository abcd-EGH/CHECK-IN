from flask import Flask, request, render_template, jsonify
import mocov2_model  # 모델 정의 모듈
from torchvision import transforms
from PIL import Image
from annoy import AnnoyIndex
import numpy as np
import sqlite3
import cv2
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import cosine_similarity
import CRAFT_pytorch.run_myCRAFT as run_myCRAFT
import torch
from shapely.geometry import Polygon
import base64
import threading
import json
from io import BytesIO

############################################ 함수 정의 ############################################

## SSIM 계산
def SSIM(path, image):
    # path: db 이미지 path
    # image: 요청한 이미지
    image_np = np.array(image)
    
    # OpenCV를 사용하여 이미지를 회색조로 변환
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    dataset_image = Image.open(path).convert("RGB")
    dataset_image = np.array(dataset_image)

    if len(dataset_image.shape) == 3 and dataset_image.shape[2] == 3:
        # Convert to grayscale if it's a 3-channel image
        dataset_image = cv2.cvtColor(dataset_image, cv2.COLOR_RGB2GRAY)

    elif len(dataset_image.shape) == 2:
        # Image is already grayscale
        dataset_image = dataset_image

    # SSIM 계산 이후 텍스트 박스 유사도를 추출하기 위해 size를 저장
    # cv2로 불러온 image의 size(shape)는 (height, width) 형태
    size = dataset_image.shape
    
    # dataset_image의 크기를 image과 동일하게 조절
    # PIL.Image 객체인 image의 size는 (width, height) 형태
    dataset_image = cv2.resize(dataset_image, (image.size[0], image.size[1]))
    ssim_index = ssim(gray_image, dataset_image)

    return ssim_index, size

## bounding box가 겹치는 영역 계산
def box(polys1, polys2, size1, size2):
    # path: 입력한 이미지 path
    # polys: 입력한 이미지 ploys
    # size1: 입력한 이미지 size
    # size2: database의 이미지 size
    # boundingbox: database의 이미지 polys
    
    # 요청한 PIL 이미지의 polys, size 계산
    polygons1 = [Polygon(poly) for poly in polys1]

    # 스케일 비율 계산
    scale_x = size1[1] / size2[1]
    scale_y = size1[0] / size2[0]

    # 입력한 이미지와 database의 이미지 사이즈가 다를 수 있음
    # 따라서, database의 이미지 polys를 입력한 이미지 size에 비례하여 스케일 조정
    polygons2 = []
    for box in polys2:
        # 스케일링된 좌표로 새로운 Polygon 생성
        scaled_box = [(x * scale_x, y * scale_y) for x, y in box]
        polygons2.append(Polygon(scaled_box))

    # 겹치는 영역 계산을 위한 초기화
    total_intersection_area = 0

    # 모든 폴리곤 조합에 대해 겹치는 영역 계산
    for p1 in polygons1:
        for p2 in polygons2:
            intersection = p1.intersection(p2)
            total_intersection_area += intersection.area

    total_area_1 = sum([p.area for p in polygons1])

    # 첫 번째 이미지를 기준으로 두 번째 이미지의 겹치는 총 면적 비율 계산
    total_overlap_percentage = (total_intersection_area / total_area_1) * 100

    return total_overlap_percentage

## 스레드 로컬 데이터베이스 연결 설정
def get_db_connection(db_path):
    if not hasattr(threading.current_thread(), 'db_connection'):
        # 스레드에 연결이 없으면 새로운 연결 생성
        threading.current_thread().db_connection = sqlite3.connect(db_path)
    return threading.current_thread().db_connection

# byte 파일을 polys array로 바꾸기
def load_polys_from_byte(byte_data):
    # 전체 폴리곤 리스트로 변환
    # 각 폴리곤은 4개의 점을 갖고, 각 점은 2개의 float32 값으로 구성됩니다.
    # 따라서, 각 폴리곤은 4 * 2 * 4 = 32 바이트를 차지합니다.
    poly_size = 4 * 2 * 4  # 한 폴리곤의 바이트 크기
    num_polys = len(byte_data) // poly_size

    polys = []
    for i in range(num_polys):
        # 각 폴리곤을 추출하고, (4, 2) 형태로 변환
        poly = np.frombuffer(byte_data[i * poly_size:(i + 1) * poly_size], dtype=np.float32).reshape(4, 2)
        polys.append(poly)

    # NumPy 배열로 변환
    return np.array(polys)


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
def encode_image_to_base64(image_input, bounding_boxes=None, box=True):
    # 입력이 파일 경로인 경우 이미지 불러오기
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
        image = np.array(image)

        if len(image.shape) == 2:
            # Convert to grayscale if it's a 3-channel image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 입력이 PIL Image 객체인 경우 해당 객체 사용
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input)
    
    # 바운딩 박스 그리기
    if box:
        image = draw_bounding_boxes(image, bounding_boxes)

    # 이미지를 PIL Image 객체로 변환
    image_pil = Image.fromarray(image)

    # 이미지를 byte로 변환 후 Base64 인코딩
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return encoded_string

############################################# 초기화 ##############################################

app = Flask(__name__)

# 모델, db RGB 채널 평균 & 표준편차 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, (mean_, std_) = mocov2_model.load_model(model_path='./mocov2_best_model_231216.pth', mean_std_path='./dataset_mean_std.pkl', train=False)

# 이미지 변환 정의
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean_, std_),
])
# Annoy 인덱스 로드
f = 128  # 특징 벡터의 차원
t = AnnoyIndex(f, 'angular')
t.load('./annoy_index_20.ann')  # 인덱스 파일이 저장된 경로로 수정

db_path = './Image_db_final.db'

######################################## html 페이지 렌더링 #######################################
@app.route('/')
def index():
    # HTML 페이지를 렌더링합니다.
    return render_template('index_final.html')

######################################## predict 함수 정의 ########################################
@app.route('/predict', methods=['POST'])
def predict():
    # 이미지 받기
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")

    # 이미지 변환 및 특징 벡터 추출
    transformed_image = test_transform(image).to(device)
    image_vector = model(transformed_image.unsqueeze(0), with_projection_head=False).detach().cpu().numpy()

    # Annoy를 사용한 유사도 검색
    # 상위 n개 결과
    n = 20
    nearest_ids = t.get_nns_by_vector(image_vector.reshape(-1), n)

    # 데이터베이스에서 해당 인덱스의 이미지 경로 조회
    similar_images = []
    # 상위 n개의 이미지에 대해 path, cos_sim, ssim_index, box_overlab_per를 list에 저장
    for idx in nearest_ids:
        cursor.execute("SELECT filename, path, feature_vector, boundingbox FROM {} WHERE id = ?".format('IMAGE'), (idx,))
        row = cursor.fetchone()
        if row:
            filename, path, feature_vector, boundingbox = row            
            # windows에서 사용하는 path에 맞게 적용
            path = path.replace('/','\\') 
            # feature_vector는 데이터베이스에서 불러온 후 적절한 형태로 변환 및 reshape
            feature_vector = np.frombuffer(feature_vector, dtype=np.float32)
            # cosine similarity 값 추출
            cos_sim = cosine_similarity(torch.tensor(image_vector), torch.tensor(feature_vector).unsqueeze(0))
            # SSIM 값, size 추출 (여기서 추출된 size2는 db_image의 shape(height, width))
            ssim_index, size2 = SSIM(path, image)
            # 텍스트 박스가 겹치는 범위, 즉 텍스트 박스 유사도(%) 추출
            # _, polys1, size1 = run_myCRAFT.return_path_poly_size(base_path='./CRAFT_pytorch/craft_mlt_25k.pth', image_PIL=image, cuda=torch.cuda.is_available())
            # if not boundingbox:
            #     image_base64 = encode_image_to_base64(path, box=False)
            #     box_overlap_per = None
            # else:
            #     polys2 = load_polys_from_byte(boundingbox)
            #     image_base64 = encode_image_to_base64(path, polys2)
            #     # box_overlap_per = box(polys1, polys2, size1, size2)
            image_base64 = encode_image_to_base64(path, box=False)
            upload_base64 = encode_image_to_base64(image, box=False)
            
            # list에 추가
            similar_images.append({
                'filename': filename,
                'path': path,
                'cos_sim': cos_sim.item(),
                'ssim_index': ssim_index,
                # 'box_overlap_per': box_overlap_per,
                'image_base64': image_base64
            })

    # cos_sim + ssim_index 값이 1.0 이상인 이미지만 저장
    similar_images = [x for x in similar_images if (x['cos_sim'] + x['ssim_index']) >= 1.0]

    # cos_sim + ssim_index + 0.1 * box_overlab_per 값을 내림차순 기준으로 sort
    similar_images = sorted(similar_images, key=lambda x: x['cos_sim'] + x['ssim_index'], reverse=True)

    # similar_images 개수가 4개 이상일 경우 4개로 조정
    if len(similar_images) > 4:
        similar_images = similar_images[:4]

    # 결과 반환
    return jsonify({'matches': similar_images, 'upload_base64': upload_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


############################################ REFERENCE ############################################
# CRAFT                                                                                           #
# Copyright (c) 2019-present NAVER Corp.                                                          #
# MIT License                                                                                     #
# CRAFT is a text detection algorithm that is used to identify text areas within an image.        #
# Reference: CRAFT GitHub Repository (https://github.com/clovaai/CRAFT-pytorch)                   #
# Research Paper: "CRAFT: Character Region Awareness for Text Detection"                          #
#                  (https://arxiv.org/abs/1904.01941)                                             #
#                                                                                                 #
# Annoy                                                                                           #
# Annoy (Approximate Nearest Neighbors Oh Yeah) - A C++ library with Python bindings              #
# to search for points in space that are close to a given query point.                            #
# Originally developed by Erik Bernhardsson at Spotify, Annoy is widely used in                   #
# various applications for efficient similarity search of high-dimensional vectors.               #
# GitHub Repository: https://github.com/spotify/annoy                                             #
# Documentation: https://github.com/spotify/annoy#readme                                          #
#                                                                                                 #
# Flask                                                                                           #
# Flask is a micro web framework for Python based on Werkzeug, Jinja 2 and good intentions.       #
# Used here to create a web server for handling image upload and processing requests.             #
# Reference: Flask Documentation (https://flask.palletsprojects.com/)                             #
#                                                                                                 #
# PyTorch                                                                                         #
# PyTorch is an open-source machine learning library for Python, used for applications such as    #
# neural networks. It is used here for loading and utilizing the deep learning model.             #
# Reference: PyTorch Documentation (https://pytorch.org/docs/stable/index.html)                   #
#                                                                                                 #
# Pillow (PIL)                                                                                    #
# Pillow is a Python Imaging Library (PIL) fork that adds image processing capabilities           #
# to your Python interpreter. In this application, it is used to open and process images.         #
# Reference: Pillow Documentation (https://pillow.readthedocs.io/en/stable/)                      #
#                                                                                                 #
# OpenCV (cv2)                                                                                    #
# OpenCV is a library of programming functions mainly aimed at real-time computer vision.         #
# Here it is used for image processing tasks such as converting images to grayscale.              #
# Reference: OpenCV Documentation (https://docs.opencv.org/master/)                               #
#                                                                                                 #
# Scikit-Image (skimage)                                                                          #
# Scikit-Image is a collection of algorithms for image processing in Python.                      #
# It is used in this application for tasks like reading images.                                   #
# Reference: Scikit-Image Documentation (https://scikit-image.org/docs/stable/)                   #
#                                                                                                 #
# SQLAlchemy                                                                                      #
# SQLAlchemy is a SQL toolkit and Object-Relational Mapping (ORM) library for Python.             #
# It is used here for database interactions.                                                      #
# Reference: SQLAlchemy Documentation (https://www.sqlalchemy.org/)                               #
#                                                                                                 #
# NumPy                                                                                           #
# NumPy is a library for the Python programming language, adding support for large,               #
# multi-dimensional arrays and matrices, along with a large collection of high-level              #
# mathematical functions to operate on these arrays.                                              #
# Reference: NumPy Documentation (https://numpy.org/doc/)                                         #
#                                                                                                 #
# EfficientNet                                                                                    #
# EfficientNet is a scalable deep learning model for computer vision,                             #
# known for its efficiency in balancing model size and accuracy. It's widely used                 #
# for tasks like image classification and object detection.                                       #
# Reference: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"           #
#             paper (https://arxiv.org/abs/1905.11946)                                            #
#                                                                                                 #
# MoCov2                                                                                          #
# MoCov2 is a deep learning model for unsupervised visual representation learning.                #
# It's effective in learning from large datasets without requiring labeled data,                  #
# improving over its predecessor MoCo with better representation quality.                         #
# Reference: "Improved Baselines with Momentum Contrastive Learning"                              #
#             paper (https://arxiv.org/abs/2003.04297)                                            #
#                                                                                                 #
# MoCo                                                                                            #
# MoCo (Momentum Contrast) is a deep learning approach for unsupervised visual                    #
# representation learning. It uses a contrastive loss framework to learn meaningful features      #
# from unlabeled images, making it useful in situations where labeled data is limited.            #
# MoCo is notable for its innovative use of a momentum-based moving average                       #
# of the model parameters, facilitating stable and effective learning of visual representations.  #
# Reference: "Momentum Contrast for Unsupervised Visual Representation Learning"                  #
#             paper (https://arxiv.org/abs/1911.05722)                                            #
#                                                                                                 #
# SQLite                                                                                          #
# SQLite is a compact, serverless SQL database engine used widely in applications                 #
# for local storage. It's known for being lightweight, requiring zero configuration,              #
# and supporting standard SQL features. SQLite is ideal for mobile apps, web browsers,            #
# and other applications where a full-scale database server is not necessary.                     #
# Reference: SQLite Official Documentation (https://www.sqlite.org/docs.html)                     #
#                                                                                                 #
# tqdm                                                                                            #
# tqdm is a Python library for displaying progress bars in the console or Jupyter notebooks.      #
# It's used to show visual progress indicators for loops and long-running processes,              #
# improving user experience and monitoring.                                                       #
# Reference: tqdm Documentation (https://tqdm.github.io/)                                         #
###################################################################################################
