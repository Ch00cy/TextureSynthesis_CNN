import numpy as np  # 과학 계산을 위한 라이브러리로서 다차원 배열을 처리하는데 필요한 여러 유용한 기능을 제공
import pdb  #  파이썬 프로그램을 위한 대화형 소스 코드 디버거

def gram_mse_loss(activations, target_gram_matrix, weight=1., linear_transform=None):
    '''
    This function computes an elementwise mean squared distance between the gram matrices of the source and the generated image.
    (이 함수는 소스의 그램 행렬과 생성된 이미지 그램 사이의 요소별 평균 제곱 거리를 계산합니다.)

    :param activations: the network activations in response to the image that is generated
        (생성된 이미지에 대한 응답으로 네트워크 활성화)
    :param target_gram_matrix: gram matrix in response to the source image
        (소스 이미지에 대한 응답으로 그램 행렬)
    :param weight: scaling factor for the loss function
        (손실 함수의 배율 인수)
    :param linear_transform: linear transform that is applied to the feature vector at all positions before gram matrix computation
        (그램 행렬 계산 전에 모든 위치에서 특징 벡터에 적용되는 선형 변환)
    :return: mean squared distance between normalised gram matrices and gradient wrt activations
        (정규화 된 그램 행렬과 그래디언트 wrt 활성화 사이의 평균 제곱 거리)
    '''

    N = activations.shape[1]    # [0]*[1] 행렬    -> 열 개수 출력
    fm_size = np.array(activations.shape[2:])
    M = np.prod(fm_size)
    G_target = target_gram_matrix
    if linear_transform == None:
        F = activations.reshape(N,-1) 
        G = np.dot(F,F.T) / M
        loss = float(weight)/4 * ((G - G_target)**2).sum() / N**2
        gradient = (weight * np.dot(F.T, (G - G_target)).T / (M * N**2)).reshape(1, N, fm_size[0], fm_size[1])
    else: 
        F = np.dot(linear_transform, activations.reshape(N,-1))
        G = np.dot(F,F.T) / M
        loss = float(weight)/4 * ((G - G_target)**2).sum() / N**2
        gradient = (weight * np.dot(linear_transform.T, np.dot(F.T, (G - G_target)).T) / (M * N**2)).reshape(1, N, fm_size[0], fm_size[1])
        
    return [loss, gradient]

def meanfm_mse_loss(activations, target_activations, weight=1., linear_transform=None):
    '''
    This function computes an elementwise mean squared distance between the mean feature maps of the source and the generated image.
    (이 함수는 소스와 생성된 이미지의 평균 기능 맵 사이의 요소별 평균 제곱 거리를 계산합니다.)

    :param activations: the network activations in response to the image that is generated
        (생성된 이미지에 대한 응답으로 네트워크 활성화)
    :param target_activations: the network activations in response to the source image
        (소스 이미지에 대한 응답으로 네트워크 활성화)
    :param weight: scaling factor for the loss function
        (손실 함수의 배율 인수)
    :param linear_transform: linear transform that is applied to the feature vector at all positions before gram matrix computation
        (그램 행렬 계산 전에 모든 위치에서 특징 벡터에 적용되는 선형 변환)
    :return: mean squared distance between mean feature maps and gradient wrt activations
        (평균 기능 맵과 그래디언트 wrt 활성화 사이의 평균 제곱 거리)
    '''

    N = activations.shape[1]
    fm_size = np.array(activations.shape[2:])
    M = np.prod(fm_size)
    
    target_fm_size = np.array(target_activations.shape[2:])
    M_target = np.prod(target_fm_size)
    if linear_transform==None:
        target_mean_fm = target_activations.reshape(N,-1).sum(1) / M_target
        mean_fm = activations.reshape(N,-1).sum(1) / M 
        f_val = float(weight)/2 * ((mean_fm - target_mean_fm)**2).sum() / N 
        f_grad = weight * (np.tile((mean_fm - target_mean_fm)[:,None],(1,M)) / (M * N)).reshape(1,N,fm_size[0],fm_size[1])
    else:
        target_mean_fm = np.dot(linear_transform, target_activations.reshape(N,-1)).sum(1) / M_target
        mean_fm = np.dot(linear_transform, activations.reshape(N,-1)).sum(1) / M 
        f_val = float(weight)/2 * ((mean_fm - target_mean_fm)**2).sum() / N 
        f_grad = weight * (np.dot(linear_transform.T ,np.tile((mean_fm - target_mean_fm)[:,None],(1,M))) / (M * N)).reshape(1,N,fm_size[0],fm_size[1])
    return [f_val,f_grad]

