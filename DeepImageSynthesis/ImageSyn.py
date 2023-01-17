import numpy as np  # import numpy as np
import matplotlib.pyplot as plt # 그림을 약간 변경
from scipy.optimize import minimize # optimize : 최적화
from Misc import *  # Misc.py (기타)

def ImageSyn(net, constraints, init=None, bounds=None, callback=None, minimize_options=None, gradient_free_region=None):
    '''
    This function generates the image by performing gradient descent on the pixels to match the constraints.
    (이 함수는 제약 조건과 일치하도록 픽셀에 경사 하강법을 수행하여 이미지를 생성합니다.)

    :param net: caffe.Classifier object that defines the network used to generate the image
        (caffe. 이미지를 생성하는 데 사용되는 네트워크를 정의하는 Classifier 객체)
    :param constraints: dictionary object that contains the constraints on each layer used for the image generation
        (이미지 생성에 사용되는 각 레이어에 대한 제약 조건을 포함하는 사전 객체)
    :param init: the initial image to start the gradient descent from. Defaults to gaussian white noise
        (경사하강법을 시작할 초기 이미지. 가우시안 화이트 노이즈 기본값)
    :param bounds: the optimisation bounds passed to the optimiser
        (옵티마이저에 전달된 최적화 범위)
    :param callback: the callback function passed to the optimiser
        (옵티마이저에 전달된 콜백 함수)
    :param minimize_options: the options passed to the optimiser
        (옵티마이저에 전달된 옵션)
    :param gradient_free_region: a binary mask that defines all pixels that should be ignored in the in the gradient descent
        (기울기 하강법에서 무시해야 하는 모든 픽셀을 정의하는 이진 마스크)
    :return: result object from the L-BFGS optimisation
        (L-BFGS 최적화의 결과 개체)
    '''

    if init==None:
        init = np.random.randn(*net.blobs['data'].data.shape)
    
     #get indices for gradient
    layers, indices = get_indices(net, constraints)
    
    #function to minimise 
    def f(x):
        x = x.reshape(*net.blobs['data'].data.shape)
        net.forward(data=x, end=layers[min(len(layers)-1, indices[0]+1)])
        f_val = 0
        #clear gradient in all layers
        for index in indices:
            net.blobs[layers[index]].diff[...] = np.zeros_like(net.blobs[layers[index]].diff)
                
        for i,index in enumerate(indices):
            layer = layers[index]
            for l,loss_function in enumerate(constraints[layer].loss_functions):
                constraints[layer].parameter_lists[l].update({'activations': net.blobs[layer].data.copy()})
                val, grad = loss_function(**constraints[layer].parameter_lists[l])
                f_val += val
                net.blobs[layer].diff[:] += grad
            #gradient wrt inactive units is 0
            net.blobs[layer].diff[(net.blobs[layer].data == 0)] = 0.
            if index == indices[-1]:
                f_grad = net.backward(start=layer)['data'].copy()
            else:        
                net.backward(start=layer, end=layers[indices[i+1]])                    

        if gradient_free_region!=None:
            f_grad[gradient_free_region==1] = 0    

        return [f_val, np.array(f_grad.ravel(), dtype=float)]            
        
    result = minimize(f, init,
                          method='L-BFGS-B', 
                          jac=True,
                          bounds=bounds,
                          callback=callback,
                          options=minimize_options)
    return result
        

