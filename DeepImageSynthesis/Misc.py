import numpy as np  # 과학 계산을 위한 라이브러리로서 다차원 배열을 처리하는데 필요한 여러 유용한 기능을 제공
import scipy    # 사이파이 : 과학기술계산을 위한 Python 라이브러리 (numpy 상위호환) -> NumPy, Matplotlib, pandas, SymPy와 연계
import caffe    #
import matplotlib.pyplot as plt # # 그림을 약간 변경
from IPython.display import display,clear_output    # Public API for display tools in IPython.
# display : 모든 프런트엔드에 Python 개체를 표시합니다. 기본적으로 모든 표현은 계산되어 프런트엔드로 전송됩니다. 프런트엔드는 사용되는 표현과 방법을 결정할 수 있습니다.
# clear_output : Clear the output of the current cell receiving output.


class constraint(object):
    '''
    Object that contains the constraints on a particular layer for the image synthesis.
    (이미지 합성을 위해 특정 레이어에 대한 제약 조건을 포함하는 개체입니다.)
    '''

    def __init__(self, loss_functions, parameter_lists):
        self.loss_functions = loss_functions
        self.parameter_lists = parameter_lists
  
def get_indices(net, constraints):
    '''
    Helper function to pick the indices of the layers included in the loss function from all layers of the network.
    (네트워크의 모든 계층에서 손실 함수에 포함된 계층의 인덱스를 선택하는 도우미 함수입니다.)
    
    :param net: caffe.Classifier object defining the network
        (네트워크를 정의하는 caffe.Classifier 객체)
    :param contraints: dictionary where each key is a layer and the corresponding entry is a constraint object
        (각 키가 레이어이고 해당 항목이 제약 조건 개체인 dictionary)
    :return: list of layers in the network and list of indices of the loss layers in descending order
        (네트워크의 계층 목록 및 내림차순 손실 계층의 인덱스 목록)
    '''

    indices = [ndx for ndx,layer in enumerate(net.blobs.keys()) if layer in constraints.keys()]
    return net.blobs.keys(),indices[::-1]

def show_progress(x, net, title=None, handle=False):
    '''
    Helper function to show intermediate results during the gradient descent.
    (경사 하강 중에 중간 결과를 표시하는 도우미 기능.)

    :param x: vectorised image on which the gradient descent is performed
        (경사하강법이 수행된 벡터화된 이미지)
    :param net: caffe.Classifier object defining the network
        (네트워크를 정의하는 caffe.Classifier 객체)
    :param title: optional title of figuer
        (그림의 선택적 제목)
    :param handle: obtional return of figure handle
        (figure handle 임의반환)
    :return: figure handle (optional)
        (figure handle (옵션))
    '''

    disp_image = (x.reshape(*net.blobs['data'].data.shape)[0].transpose(1,2,0)[:,:,::-1]-x.min())/(x.max()-x.min())
    clear_output()
    plt.imshow(disp_image)
    if title != None:
        ax = plt.gca()
        ax.set_title(title)
    f = plt.gcf()
    display()
    plt.show()    
    if handle:
        return f
   
def get_bounds(images, im_size):
    '''
    Helper function to get optimisation bounds from source image.
    (소스 이미지에서 최적화 범위를 가져오는 도우미 함수입니다.)

    :param images: a list of images
        (이미지 목록)
    :param im_size: image size (height, width) for the generated image
        (생성된 이미지의 이미지 크기(높이, 너비))
    :return: list of bounds on each pixel for the optimisation
        (최적화를 위한 각 픽셀의 범위 목록)
    '''

    lowerbound = np.min([im.min() for im in images])
    upperbound = np.max([im.max() for im in images])
    bounds = list()
    for b in range(im_size[0]*im_size[1] * 3):
        bounds.append((lowerbound,upperbound))
    return bounds 

def test_gradient(function, parameters, eps=1e-6):
    '''
    Simple gradient test for any loss function defined on layer output
    (레이어 출력에 정의된 모든 손실 함수에 대한 간단한 그래디언트 테스트)

    :param function: function to be tested, must return function value and gradient
        (테스트할 함수는 함수 값과 그래디언트를 반환해야 합니다.)
    :param parameters: input arguments to function passed as keyword arguments
        (키워드 인수로 전달된 함수에 대한 입력 인수)
    :param eps: step size for numerical gradient evaluation
        (수치 기울기 평가를 위한 단계 크기)
    :return: numerical gradient and gradient from function
        (함수의 수치 기울기 및 기울기)
    '''

    i,j,k,l = [np.random.randint(s) for s in parameters['activations'].shape]
    f1,_ = function(**parameters)
    parameters['activations'][i,j,k,l] += eps
    f2,g = function(**parameters)
    
    return [(f2-f1)/eps,g[i,j,k,l]]

def gram_matrix(activations):
    '''
    Gives the gram matrix for feature map activations in caffe format with batchsize 1. Normalises by spatial dimensions.
    (배치 크기가 1인 caffe 형식의 기능 맵 활성화에 대한 그램 매트릭스를 제공합니다. 공간 차원으로 정규화합니다.)

    :param activations: feature map activations to compute gram matrix from
        (그람 행렬을 계산하기 위한 기능 맵 활성화)
    :return: normalised gram matrix
        (정규화 그램 매트릭스)
    '''

    N = activations.shape[1]
    F = activations.reshape(N,-1)
    M = F.shape[1]
    G = np.dot(F,F.T) / M
    return G
    
def disp_img(img):
    '''
    Returns rescaled image for display with imshow
    (imshow로 표시하기 위해 크기 조정된 이미지를 반환합니다.)
    '''
    disp_img = (img - img.min())/(img.max()-img.min())
    return disp_img  

def uniform_hist(X):
    '''
    Maps data distribution onto uniform histogram
    (균일한 히스토그램에 데이터 분포 매핑)
    
    :param X: data vector
        (벡터 데이터)
    :return: data vector with uniform histogram
        (히스토그램이 균일한 데이터 벡터)
    '''

    Z = [(x, i) for i, x in enumerate(X)]   # 인덱스와 원소로 이루어진 튜플(tuple) 만듬 -> i : 인덱스(0부터 ~) , x : X의 원소
    Z.sort()    # 메소드만 가능 , 오름차순 정렬
    n = len(Z)
    Rx = [0]*n
    start = 0 # starting mark
    for i in range(1, n):
        if Z[i][0] != Z[i-1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start+1+i)/2.0;
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start+1+n)/2.0;
    return np.asarray(Rx) / float(len(Rx))

def histogram_matching(org_image, match_image, grey=False, n_bins=100):
    '''
    Matches histogram of each color channel of org_image with histogram of match_image
    (org_image의 각 색상 채널의 히스토그램을 match_image의 히스토그램과 일치시킵니다.)

    :param org_image: image whose distribution should be remapped
        (분포를 다시 매핑해야 하는 이미지)
    :param match_image: image whose distribution should be matched
        (분포가 일치해야 하는 이미지)
    :param grey: True if images are greyscale
        (이미지가 회색조인 경우 참)
    :param n_bins: number of bins used for histogram calculation
        (히스토그램 계산에 사용되는 bins 수)
    :return: org_image with same histogram as match_image
        (match_image와 동일한 히스토그램이 있는 org_image)
    '''

    if grey:
        hist, bin_edges = np.histogram(match_image.ravel(), bins=n_bins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
        inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
        r = np.asarray(uniform_hist(org_image.ravel()))
        r[r>cum_values.max()] = cum_values.max()    
        matched_image = inv_cdf(r).reshape(org_image.shape) 
    else:
        matched_image = np.zeros_like(org_image)
        for i in range(3):
            hist, bin_edges = np.histogram(match_image[:,:,i].ravel(), bins=n_bins, density=True)
            cum_values = np.zeros(bin_edges.shape)
            cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
            inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges,bounds_error=True)
            r = np.asarray(uniform_hist(org_image[:,:,i].ravel()))
            r[r>cum_values.max()] = cum_values.max()    
            matched_image[:,:,i] = inv_cdf(r).reshape(org_image[:,:,i].shape)
        
    return matched_image

def load_image(file_name, im_size, net_model, net_weights, mean, show_img=False):
    '''
    Loads and preprocesses image into caffe format by constructing and using the appropriate network.
    (적절한 네트워크를 구성하고 사용하여 이미지를 caffe 형식으로 로드하고 전처리합니다.)

    :param file_name: file name of the image to be loaded
        (로드할 이미지의 파일 이름)
    :param im_size: size of the image after preprocessing if float that the original image is rescaled to contain im_size**2 pixels
        (원본 이미지가 im_size**2 픽셀을 포함하도록 재조정되는 경우 전처리 후 이미지 크기)
    :param net_model: file name of the prototxt file defining the network model
        (네트워크 모델을 정의하는 prototxt 파일의 파일 이름)
    :param net_weights: file name of caffemodel file defining the network weights
        (네트워크 가중치를 정의하는 caffemodel 파일의 파일 이름)
    :param mean: mean values for each color channel (bgr) which are subtracted during preprocessing
        (전처리 중에 빼는 각 색상 채널(bgr)의 평균값)
    :param show_img: if True shows the loaded image before preprocessing
        (True가 전처리 전에 로드된 이미지를 표시하는 경우)
    :return: preprocessed image and caffe.Classifier object defining the network
        (네트워크를 정의하는 전처리된 이미지 및 caffe.Classifier 객체)
    '''

    img = caffe.io.load_image(file_name)
    if show_img:
        plt.imshow(img)
    if isinstance(im_size,float):
        im_scale = np.sqrt(im_size**2 /np.prod(np.asarray(img.shape[:2])))
        im_size = im_scale * np.asarray(img.shape[:2])
    batchSize = 1
    with open(net_model,'r+') as f:
        data = f.readlines() 
    data[2] = "input_dim: %i\n" %(batchSize)
    data[4] = "input_dim: %i\n" %(im_size[0])
    data[5] = "input_dim: %i\n" %(im_size[1])
    with open(net_model,'r+') as f:
        f.writelines(data)
    net_mean =  np.tile(mean[:,None,None],(1,) + tuple(im_size.astype(int)))
    #load pretrained network
    net = caffe.Classifier( 
    net_model, net_weights,
    mean = net_mean,
    channel_swap=(2,1,0),
    input_scale=255,)
    img_pp = net.transformer.preprocess('data',img)[None,:]
    return[img_pp, net]
