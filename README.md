# KCC_EGG

## Dataset download link

    https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format


## 1. 특정 필터 사용 안하는 방법
#### 1.1. 필터의 gradient zero
    gradient update 이전에 필터가 학습에 사용 되었기 때문에 학습에 영향을 줄 수 밖에 없다.
    또한 FC layer의 파라미터 값들은 변경 되기 때문에 기존의 모델을 해치게 된다.
    => FC Layer 파라미터 값도 고정 시켜놓고 학습해도 모델의 성능 저하.
    
TODO.

    1. class별 정확도 측정
#### 1.2. 필터를 완전 detach 하거나 끄는 방법. => 찾아봐야함
#### 1.3. CNN필터를 하나씩 선언해서 사용하는 방법 => 노가다가 필요.

