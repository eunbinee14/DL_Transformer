# DL_Transformer

- Transformer operation

<br/><br/>

## 1. Development Environment Assign
- Python version: 3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)] 
- PyTorch version: 1.8.1+cpu 
- TorchText version: 0.9.1 
- CUDA version: 11.8
- Numpy version : 1.22.0

<br/><br/>
## 2. pipeline
![스크린샷 2024-12-30 002905](https://github.com/user-attachments/assets/42351713-29b7-497e-baea-4961430eba1a)

<br/><br/>

## 3. logic & main class
### 3.1 tokenizer
- 영어 encoder 토큰화
- 독일어 decoder 토큰화

### 3.2 token embedding
- 고정되지 않는 임베딩 벡터
- 학습 과정 중 loss 최소화 업데이트
- 문맥과 의미를 반영하도록 업데이트

### 3.3 positional encoding
- Transformer 모델은 순서가 없는 구조, 순서 정보를 추가하기 위해 이용
- self-attention 매커니즘 = 순서가 없음
- 단어의 순서, 위치 정보를 위해 이용
- 위치 정보 
    - 짝수 차원 = sin 함수, 홀수 차원 = cos 함수

### 3.4 scale dot product attention
- self-attention 매커니즘, 문맥 정보 얻음
- 각 단어가 다른 단어와의 연관성 파악
- multi-head attention에서 만들어진 Q, K, V 전달됨
- Q, K 유사도 계산 후 softmax 통과하여 확률 값과 V를 곱하여 output return

### 3.5 multi-head attention
- X(입력데이터)가 학습 가능한 가중치를 통하여 Q, K, V 생성
- 각각 n.head로 나누어서 scale dot proudct attention 병렬적 계산 실행
- output head 출력 병합(concat)
- nn.Linear(선형 변환) 최종 출력

### 3.6 positionwise feed forward
- multi-head attention의 출력에 대해 각 단어의 특징을 독립적으로 변환
- nn.Linear & ReLU : 비선형적 변환
- self.linear1
    - 입력 벡터 확장
    - d_model을 hidden으로
- self.linear2
    - 원래 차원으로 벡터 축소
    - hidden을 d_model로

### 3.7 transformer
- Encoder와 Decoder를 통합한 모델의 전체 구조를 구현


<br/><br/>
## 4. Reference
source code : https://github.com/gusdnd852
