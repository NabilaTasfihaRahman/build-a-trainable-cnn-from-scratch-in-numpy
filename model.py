"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
import numpy as np
def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array
    
    output=np.max(matrix,axis=1)
    index=[]
    for i in range(matrix.shape[0]):
        ind=np.where(output[i]==matrix[i,:])[0][0]
        index.append(ind)
    return np.array(index)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.
    return np.max(matrix,axis=1,keepdims=True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    output=np.sum(matrix,axis=1,keepdims=True)
    return output

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    max=np.max(logits,axis=1,keepdims=True)
    logits_new=np.exp(logits-max)
    return logits_new

# Step 5 - stable_softmax
def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    logit_shifted=exp_shifted(logits)
    output=logit_shifted/row_sum(logit_shifted)
    return output

# Step 6 - one_hot
import numpy as np
def one_hot(labels, num_classes):
    # TODO: convert integer labels into a (N, num_classes) one-hot float matrix
    # newlabels,idx=np.unique(labels,return_index=True)
    # blabels=[labels[i] for i in sorted(idx)]
    new_label=np.zeros((labels.shape[0],num_classes))
    # if labels.shape[0]==num_classes:
    for i in range(new_label.shape[0]):
    
        new_label[i,labels[i]]=1.0
    

    return new_label

# Step 7 - gather_true_class_probs
import numpy as np
def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    listi=[]
    for i in range(probs.shape[0]):
        listi.append(probs[i,labels[i]])
    return np.array(listi)

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    true_prob=gather_true_class_probs(probs,labels)
    true_prob= true_prob+eps
    return np.mean(-np.log(true_prob))

# Step 9 - accuracy
import numpy as np
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    outputs=argmax_rows(logits_or_probs)
    accuracy=np.mean(outputs==labels)
    return accuracy

# Step 10 - he_std
import numpy as np
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return np.sqrt(2/fan_in)

# Step 11 - he_init
import numpy as np
def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.
    np.random.seed(seed)
    weight=np.random.normal(0,he_std(fan_in),shape)
    return np.array(weight)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros((length,),dtype=float)

# Step 13 - pad_2d
import numpy as np
def pad_2d(images, pad):
    # TODO: zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.
    if pad==0:
        return images
    else:
        new=np.pad(images,pad_width=
        ((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')
        return new

# Step 14 - output_spatial_size
import numpy as np
def output_spatial_size(input_size, kernel, stride, padding):
    # TODO: return the conv/pool output spatial dimension from input_size, kernel, stride, padding
    return int(np.floor((input_size+2*padding-kernel)/stride)+1)

# Step 15 - im2col
import numpy as np
def im2col(images, kernel_h, kernel_w, stride, padding):
    # TODO: Unroll overlapping patches of a 4D image tensor into a 2D column matrix.
    image_new=pad_2d(images,padding)
    N,C,H,W=image_new.shape
    out_h=output_spatial_size(images.shape[2],kernel_h,stride,padding)
    out_w=output_spatial_size(images.shape[3],kernel_w,stride,padding)
    bleh=0
    output=np.zeros((N*out_h*out_w,C*kernel_h*kernel_w))
    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                start_h=stride*i
                start_w=stride*j
                patch=image_new[n,:,
                start_h:start_h+kernel_h,
                start_w:start_w+kernel_w]
                output[bleh]=patch.reshape(-1)
                bleh=bleh+1
    return output.astype(images.dtype)

# Step 16 - col2im
import numpy as np
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # TODO: re-roll a (N*out_h*out_w, C*kh*kw) column matrix back into a (N, C, H, W) tensor
    row=cols.shape[0]
    col=cols.shape[1]
    N,C,H,W=input_shape
    out_h=output_spatial_size(H,kernel_h,stride,padding)
    out_w=output_spatial_size(W,kernel_h,stride,padding)
    image=np.zeros((N,C,H+2*padding,W+2*padding))
    count=np.zeros((N,C,H+2*padding,W+2*padding))
    row_idx=0
    for n in range(N):
        for i in range(out_h):
            
            for j in range(out_w):
                patch=cols[row_idx].reshape(-1,kernel_h,kernel_w)
                h_start=i*stride
                w_start=j*stride
                image[n,:,h_start:h_start+kernel_h,w_start:w_start+kernel_w]+=patch
                count[n,:,h_start:h_start+kernel_h,w_start:w_start+kernel_w]+=1
                row_idx+=1
    if padding>0:
        return image[:,:,padding:-padding,padding:-padding]
    else:
        return image

# Step 17 - conv2d_forward
import numpy as np
def conv2d_forward(x, weights, bias, stride, padding):
    # TODO: convolve x with weights using im2col, add bias, return output and a backprop cache.
    
    Cout,_,kh,kw=weights.shape
    N,_,H,W=x.shape
    hnew=output_spatial_size(H,kh,stride,padding)
    wnew=output_spatial_size(W,kw,stride,padding)
    xcols=im2col(x,kh,kw,stride,padding).astype(x.dtype)
    dictionary={'x_shape':x.shape,
    'weights':weights,
    'bias':bias,
    'stride':stride,
    'padding':padding,
    'cols': xcols,
    'kernel_h':kh,
    'kernel_w':kw
    }
    
    wflat=np.reshape(weights,(weights.shape[0],-1))
    result=np.matmul(xcols,np.transpose(wflat))+bias
    return result.reshape(N, hnew, wnew, Cout).transpose(0, 3, 1, 2),dictionary
    # return dictionary

# Step 18 - conv2d_grad_input
import numpy as np
def conv2d_grad_input(d_out, cache):
    # TODO: backprop d_out through the conv input using col2im
    inputshape=cache['x_shape']
    stride=cache['stride']
    padding=cache['padding']
    kernel_h=cache['kernel_h']
    kernel_w=cache['kernel_w']
    d_out=np.transpose(d_out,(0,2,3,1))
    weights=cache["weights"]

    weights_2d=np.reshape(weights,(weights.shape[0],-1))
    dout_2d=np.reshape(d_out,(-1,d_out.shape[-1]))
    grad_input=np.matmul(dout_2d,weights_2d)
    return col2im(grad_input,inputshape,kernel_h,kernel_w,stride,padding)

# Step 19 - conv2d_grad_weights
import numpy as np
def conv2d_grad_weights(d_out, cache):
    # TODO: return dL/dW shaped (C_out, C_in, kH, kW) from d_out and the im2col cache.
    
    import numpy as np

    # TODO: backprop d_out through the conv input using col2im
    
    # stride=cache['stride']
    # padding=cache['padding']
    kernel_h=cache['kernel_h']
    kernel_w=cache['kernel_w']
    x=cache['cols']
    d_out=np.transpose(d_out,(0,2,3,1))
    weights=cache["weights"]
    weightshape=weights.shape
    x_2d=np.reshape(x,(x.shape[0],-1))
    dout_2d=np.reshape(d_out,(-1,d_out.shape[-1]))
    grad_input=np.matmul(np.transpose(dout_2d),x)
    return np.reshape(grad_input,weightshape)

# Step 20 - conv2d_grad_bias
import numpy as np
def conv2d_grad_bias(d_out):
    # TODO: return a length C_out gradient by reducing d_out over batch and spatial axes
    d_out=np.transpose(d_out,(0,2,3,1))
    return np.sum(d_out,axis=(0,1,2))

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    # TODO: return (dx, dW, db) using the conv2d gradient helpers and the forward cache
    dx=conv2d_grad_input(d_out,cache)
    dW=conv2d_grad_weights(d_out,cache)
    db=conv2d_grad_bias(d_out)
    return (dx,dW,db)

# Step 22 - maxpool2d_forward
import numpy as np
def maxpool2d_forward(x, kernel, stride):
    # TODO: run 2D max pooling and cache the in-window argmax of each output cell.
    N,C,h,w=x.shape
    outputsize=output_spatial_size(h,kernel,stride,0)
    
    row_start=0
    col_start=0
    row_end=kernel
    col_end=kernel
    out=np.zeros((N,C,outputsize,outputsize))
    argmax=np.zeros_like(out,dtype=int)
    for i in range(outputsize):
        for j in range(outputsize):
            out[:,:,i,j]=np.max(x[:,:,row_start:row_end,col_start:col_end],axis=(2,3))
            window=x[:,:,row_start:row_end,col_start:col_end]
            reshape_x=window.reshape(N,C,kernel*kernel)
            argmax[:,:,i,j]=np.argmax(reshape_x,2)
            col_start+=stride
            col_end+=stride
        row_start+=stride
        row_end+=stride
        col_start=0
        col_end=kernel
    cache={"x_shape":x.shape,
    "argmax":argmax,
    'kernel':kernel,
    "stride":stride}
    return (out,cache)

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    # TODO: place grad_value at the argmax position within a (kernel, kernel) zero array.
    row=argmax_index//kernel
    col=argmax_index%kernel
    kernel_matrix=np.zeros((kernel,kernel))
    kernel_matrix[row,col]=grad_value
    return kernel_matrix

# Step 24 - maxpool2d_backward
import numpy as np
def maxpool2d_backward(d_out, cache):
    # TODO: scatter each d_out value to the cached argmax position in its window
    x_shape=cache['x_shape']
    argmax=cache['argmax']
    kernel=cache['kernel']
    stride=cache['stride']
    N,C,h,w=x_shape
    grad=np.zeros((N,C,h,w))
    N,C,outh,outw=d_out.shape
    for p in range(N):
        for q in range(C):
            for i in range(outh):
                for j in range(outw):
                    row=argmax[p,q,i,j]//kernel
                    col=argmax[p,q,i,j]%kernel
                    grad[p,q,i*stride+row,j*stride+col]+=d_out[p,q,i,j]
    return grad

# Step 25 - relu_forward
import numpy as np
def relu_forward(x):
    # TODO: Compute the elementwise ReLU and cache the input for backprop.
    cache={"x":x}
    out=np.maximum(x,0)
    return (out,cache)

# Step 26 - relu_backward
import numpy as np
def relu_backward(d_out, cache):
    # TODO: mask the upstream gradient by the positive entries of the cached input.
    x=cache['x']
    dim=x.ndim

    output=np.where(x<=0)
    d_out[output]=0
    return d_out

# Step 27 - flatten_forward
import numpy as np
def flatten_forward(x):
    # TODO: reshape a 4D feature map into a 2D batch matrix and cache the original shape
    cache={'x_shape':x.shape}
    return (np.reshape(x,(x.shape[0],-1)),cache)

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    # TODO: reshape the upstream gradient back to the original 4D feature map shape.
    shape=cache['x_shape']
    return np.reshape(d_out,shape)

# Step 29 - linear_forward
import numpy as np
def linear_forward(x, weights, bias):
    # TODO: compute X @ W + b and cache the inputs needed for backprop.
    cache={'x':x,
    'weights':weights}
    return np.matmul(x,weights)+bias ,cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    """Gradient of a linear layer w.r.t. its input X."""
    # TODO: return dL/dX given d_out (N, D_out) and cache['weights'] (D_in, D_out)
    weights=cache['weights']
    grad_input=np.matmul(d_out,np.transpose(weights))
    return grad_input

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""
    # TODO: Compute the gradient of a linear layer's loss wrt its weight matrix W.
    
    grad_w=np.matmul(np.transpose(x),dout)
    return grad_w

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    # TODO: Compute the bias gradient of a linear layer given upstream gradient dout.
    return np.sum(dout,axis=0)

# Step 33 - linear_backward
def linear_backward(dout, cache):
    # TODO: combine input, weight, and bias gradients for a linear layer using the cache
    dx=linear_grad_input(dout,cache)
    x=cache['x']
    dw=linear_grad_weights(x,dout)
    db=linear_grad_bias(dout)
    return (dx,dw,db)

# Step 34 - softmax_cross_entropy_forward
import numpy as np
def softmax_cross_entropy_forward(logits, y):
    # TODO: return the mean cross-entropy loss for logits (N, C) and integer labels y (N,).
    probs=stable_softmax(logits)
    return max(cross_entropy_loss(probs,y,1e-12),0.0)

# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    # TODO: return the fused softmax-cross-entropy gradient of shape (N, C).
    N=logits.shape[0]
    C=logits.shape[1]
    onehot=one_hot(y,C)
    return ((stable_softmax(logits)-onehot)/N)

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    # TODO: return the SGD-updated parameter array (param - lr * grad).
    newparam=param-lr*grad
    return newparam

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    # TODO: return the updated first moment estimate using beta_one and grad.
    updated_moment=beta_one*m+(1-beta_one)*grad
    return updated_moment

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    # TODO: return the updated Adam second moment estimate as an EMA of squared gradients.
    return beta_two*v+(1-beta_two)*grad**2

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.
    return moment/(1-beta**t)

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
    # TODO: apply one Adam parameter update using bias-corrected moments
    new_param=param-lr*m_hat/(np.sqrt(v_hat)+eps)
    return new_param

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    # TODO: chain the four Adam helpers and return (new_param, new_m, new_v)
    m=adam_update_m(m,grad,beta_one)
    v=adam_update_v(v,grad,beta_two)
    m_hat=adam_bias_correct(m,beta_one,t)
    v_hat=adam_bias_correct(v,beta_two,t)
    return adam_param_step(param,m_hat, v_hat,lr,eps),m,v

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    # TODO: Build He-initialized weights and a zero bias for a single conv layer.
    shape=(out_channels,in_channels,kernel_size,kernel_size)
    w=he_init(shape,in_channels*kernel_size*kernel_size,seed)
    b=init_zero_bias(out_channels)
    dictionary={'W':w,
    'b':b}
    return dictionary

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    # TODO: return {'W': He-init matrix (in_features, out_features), 'b': zero bias (out_features,)}
    return {'W': he_init((in_features, out_features),in_features,seed), 
    'b': init_zero_bias(out_features,)}

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    # TODO: build conv1, conv2, fc1, fc2 with the right shapes and return them in a dict.
    conv1=init_conv_layer(6,in_channels,5,seed=0)
    conv2=init_conv_layer(16,6,5,seed=1)
    fc1=init_linear_layer(256,120,seed=2)
    fc2=init_linear_layer(120,num_classes,seed=3)
    return {'conv1':{'W':conv1['W'],'b':conv1['b']},
    'conv2':{'W':conv2['W'],'b':conv2['b']},
    'fc1':{'W':fc1['W'],'b':fc1['b']},
    'fc2':{'W':fc2['W'],'b':fc2['b']}}

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, pad):
    # TODO: run conv2d -> relu -> maxpool2d and return (out, cache_dict)
   conv,conv_cache=conv2d_forward(x,W,b,stride,pad)
   relu,relu_cache=relu_forward(conv)
   out,pool_cache=maxpool2d_forward(relu,pool_size,pool_size)
   cache={'conv_cache':conv_cache,
   'relu_cache':relu_cache,
   'pool_cache':pool_cache}
   return out,cache

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    # TODO: run flatten -> linear -> relu -> linear and return logits plus a cache dict.
    flatten,flatten_cache=flatten_forward(x)
    fc1,fc1_cache=linear_forward(flatten,fc1['W'],fc1['b'])
    relu,relu_cache=relu_forward(fc1)
    logits,fc2_cache=linear_forward(relu,fc2['W'],fc2['b'])
    return logits,{'flatten_cache':flatten_cache,
    'fc1_cache':fc1_cache,
    'relu_cache':relu_cache,
    'fc2_cache':fc2_cache}

# Step 47 - lenet_forward
def lenet_forward(x, params):
    # TODO: run two conv blocks then the classifier block and return (logits, caches).
    
    W1=params['conv1']['W']
    b1=params['conv1']['b']
    W2=params['conv2']['W']
    b2=params['conv2']['b']
    fc1=params['fc1']
    fc2=params['fc2']
    conv1,block1_cache=forward_conv_block(x,W1,b1,2,1,0)
    conv2,block2_cache=forward_conv_block(conv1,W2,b2,2,1,0)
    logit,classifier_cache=forward_classifier_block(conv2,fc1,fc2)
    return logit,{'block1':block1_cache,
    'block2':block2_cache,
    'classifier':classifier_cache}

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    # TODO: backprop dout through the cached pool, relu, and conv layers in reverse order.
    conv_cache=cache['conv_cache']
    pool_cache=cache["pool_cache"]
    relu_cache=cache['relu_cache']
    dp=maxpool2d_backward(dout,pool_cache)
    drelu=relu_backward(dp,relu_cache)
    return conv2d_backward(drelu,conv_cache)

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    # TODO: backprop through fc2 -> relu -> fc1 -> flatten using the cached values
    fc1_cache=cache['fc1_cache']
    fc2_cache=cache['fc2_cache']
    
    flatten_cache=cache['flatten_cache']
    relu_cache=cache['relu_cache']
    out,dw2,db2=linear_backward(dlogits,fc2_cache)
    relu=relu_backward(out,relu_cache)
    fc1,dw1,db1=linear_backward(relu,fc1_cache)
    flatten=flatten_backward(fc1,flatten_cache)
    return {'fc2':{'dW':dw2,'db':db2},
    'fc1':{'dW':dw1,'db':db1},
    'dx':flatten}

# Step 50 - lenet_backward
def lenet_backward(dlogits, caches):
    # TODO: walk classifier and conv block caches in reverse to assemble all gradients
    classifier=caches['classifier']
    
    block2=caches['block2']
    block1=caches['block1']
    
    back_classifier=backward_classifier_block(dlogits,classifier)
    back_conv2=backward_conv_block(back_classifier['dx'],block2)
    back_conv1=backward_conv_block(back_conv2[0],block1)
    return {
    'conv1':{'dW':back_conv1[1],'db':back_conv1[2]},
    'conv2':{'dW':back_conv2[1],'db':back_conv2[2]},
    'fc1':{'dW':back_classifier['fc1']['dW'],'db':back_classifier['fc1']['db']},
    'fc2':{'dW':back_classifier['fc2']['dW'],'db':back_classifier['fc2']['db']}

    }

# Step 51 - lenet_predict
def lenet_predict(x, params):
    # TODO: Return the argmax class index per sample from a LeNet forward pass.
    output,_=lenet_forward(x,params)
    return argmax_rows(output)

# Step 52 - build_synthetic_image_dataset
import numpy as np
def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    # TODO: Return (x, y) for a reproducible synthetic NCHW image dataset.
    
    generator=np.random.default_rng(seed)
    labels=generator.integers(0,num_classes,num_samples)
    x=generator.standard_normal((num_samples,in_channels,image_size,image_size))

    return (x,labels)

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
    # TODO: return a reproducible permutation of [0, n) as an int ndarray of shape (n,).
    np.random.seed(seed)
    idx=np.random.permutation(n)
    return idx

# Step 54 - train_test_split
import numpy as np
def train_test_split(x, y, test_fraction=0.2, seed=0):
    # TODO: partition x and y into train and test halves using a shared shuffled order.
    n=y.shape[0]
    idx=shuffle_indices(n,seed)
    test_fraction=int(n*test_fraction)
    x_test=x[idx[:test_fraction],:]
    x_train=x[idx[test_fraction:],:]
    y_test=y[idx[:test_fraction]]
    y_train=y[idx[test_fraction:]]
    return (x_train, y_train, x_test, y_test)

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

