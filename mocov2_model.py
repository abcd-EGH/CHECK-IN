# EfficientNet-PyTorch (참조: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", Tan and Le, 2019)

# 아래는 MoCo v2에 대한 구현입니다. MoCo는 "Momentum Contrast for Unsupervised Visual Representation Learning" (He et al., 2020)에서 처음 소개되었습니다.
# MoCo v2는 "Improved Baselines with Momentum Contrastive Learning" (Chen et al., 2020)에서 소개되었습니다.
# 이 구현은 원 논문의 접근 방식을 기반으로 하며, 몇 가지 수정사항이 포함되어 있습니다.

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from efficientnet_pytorch import EfficientNet
import copy
import pickle

# epoch 및 hyperparameter 정의
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the MoCov2 model
class MoCov2Model(nn.Module):
    def __init__(self, base_encoder, projection_dim=128, temperature=0.07, dropout_rate=0.5, queue_size=384, momentum=0.99):
        super(MoCov2Model, self).__init__()

        # 이미지 특징 추출 - 학습 & 추론 단계에서 모두 사용
        # EfficientNet + 128차원의 출력을 가진 선형 레이어 추가하여 DB에 저장할 특징 벡터의 차원 수 축소
        self.encoder = nn.Sequential(
            base_encoder,
            nn.Linear(1000, 128)
        )

        # Encoder에 의해 추출된 특징을 projection - 학습 과정에서 contrastive loss를 계산하는 데 사용, 추론 시엔 사용 X
        # Projection head 추가
        self.projection_head = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
            nn.Dropout(dropout_rate)
        )

        # 모멘텀 인코더의 복사본 생성 (깊은 복사)
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.momentum_projection_head = copy.deepcopy(self.projection_head)

        # 모멘텀 인코더와 projection head의 모든 파라미터를 고정 (훈련 중 업데이트 방지)
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        for param in self.momentum_projection_head.parameters():
            param.requires_grad = False

        # 기타 초기화
        self.queue = torch.zeros(queue_size, projection_dim).to(device)
        self.queue_ptr = 0
        self.momentum = momentum

    def _momentum_update(self):
        # 모멘텀 인코더 및 projection head의 가중치를 업데이트하는 함수
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            for param_q, param_k in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward(self, x, with_projection_head=True):
        x = self.encoder(x)
        if with_projection_head:
            x = self.projection_head(x)
        return x

    def enqueue_dequeue(self, keys):
        # 큐에 새로운 데이터 추가 및 오래된 데이터 제거 (CPU에서 수행)
        keys = keys.to('cpu')  # GPU에서 CPU로 이동
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        space_left = self.queue.size(0) - ptr  # 남은 공간 계산
        # 마지막 배치의 크기가 다른 배치보다 작을 때 발생할 수 있는 예외 상황 처리
        if space_left < batch_size:
            # If 큐의 남은 공간이 batch_size보다 작을 경우, split the update
            self.queue[ptr:] = keys[:space_left]
            self.queue[:batch_size - space_left] = keys[space_left:]
            ptr = batch_size - space_left
        else:
            self.queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.queue.size(0)  # move pointer

        self.queue_ptr = ptr


def load_model(model_path='./mocov2_best_model_231216.pth', mean_std_path='./dataset_mean_std.pkl', train=False):
    with open(mean_std_path, 'rb') as f:
        mean_, std_ = pickle.load(f)

    efficientnet_b2 = EfficientNet.from_name('efficientnet-b2').to(device)
    efficientnet_b2.load_state_dict(torch.load('.\\efficientnet-b2.pth'))
    mocov2_model = MoCov2Model(efficientnet_b2).to(device)

    if torch.cuda.is_available():
        mocov2_model.load_state_dict(torch.load(model_path))
    else:
        mocov2_model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    if train:
        mocov2_model.train()
    else:
        mocov2_model.eval()
    
    return mocov2_model, (mean_, std_)