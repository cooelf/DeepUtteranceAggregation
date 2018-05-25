from __future__ import print_function

import cPickle
import numpy as np
import theano
from gensim.models.word2vec import Word2Vec
import theano.tensor as T
from theano.tensor.nnet import conv
import scipy.sparse as sp
from collections import defaultdict, OrderedDict
import sys, re, random, logging, argparse
import datetime


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')

# weight initializer, normal by default
def norm_weight(nin, nout=None, scale=0.01, ortho=False):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')

def uniform_weight(size,scale=0.1):
    return np.random.uniform(size=size,low=-scale, high=scale).astype(theano.config.floatX)

def glorot_uniform(size):
    fan_in, fan_out = size
    s = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(size=size,low=-s, high=s).astype(theano.config.floatX)

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

def kmaxpooling(input,input_shape,k):
    sorted_values = T.argsort(input,axis=3)
    topmax_indexes = sorted_values[:,:,:,-k:]
    # sort indexes so that we keep the correct order within the sentence
    topmax_indexes_sorted = T.sort(topmax_indexes)

    #given that topmax only gives the index of the third dimension, we need to generate the other 3 dimensions
    dim0 = T.arange(0,input_shape[0]).repeat(input_shape[1]*input_shape[2]*k)
    dim1 = T.arange(0,input_shape[1]).repeat(k*input_shape[2]).reshape((1,-1)).repeat(input_shape[0],axis=0).flatten()
    dim2 = T.arange(0,input_shape[2]).repeat(k).reshape((1,-1)).repeat(input_shape[0]*input_shape[1],axis=0).flatten()
    dim3 = topmax_indexes_sorted.flatten()
    return input[dim0,dim1,dim2,dim3].reshape((input_shape[0], input_shape[1], input_shape[2], k))

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


class GRU(object):
    def __init__(self, n_in, n_hidden, n_out, activation=T.tanh,inner_activation=T.nnet.sigmoid,
                 output_type='real',batch_size=200):

        self.activation = activation
        self.inner_activation = inner_activation
        self.output_type = output_type

        self.batch_size = batch_size
        self.n_hidden = n_hidden

        # recurrent weights as a shared variable
        self.U_z = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.W_z = theano.shared(glorot_uniform((n_in,n_hidden)),borrow=True)
        self.b_z = theano.shared(value=np.zeros((n_hidden,),dtype=theano.config.floatX),borrow=True)

        self.U_r = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.W_r = theano.shared(glorot_uniform((n_in,n_hidden)),borrow=True)
        self.b_r = theano.shared(value=np.zeros((n_hidden,),dtype=theano.config.floatX),borrow=True)

        self.U_h = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.W_h = theano.shared(glorot_uniform((n_in,n_hidden)),borrow=True)
        self.b_h = theano.shared(value=np.zeros((n_hidden,),dtype=theano.config.floatX),borrow=True)


        self.params = [self.W_z,self.W_h,self.W_r,
                       self.U_h,self.U_r,self.U_z,
                       self.b_h,self.b_r,self.b_z]

    def __call__(self, input,input_lm=None, return_list = False, return_list_except_last = False, Init_input =None,check_gate = False):
         # activation function
        if Init_input == None:
            init = theano.shared(value=np.zeros((self.batch_size,self.n_hidden),
                                                                  dtype=theano.config.floatX),borrow=True)
        else:
            init = Init_input

        if check_gate:
            self.h_l, _ = theano.scan(self.step3,
                        sequences=[input.dimshuffle(1,0,2),T.addbroadcast(input_lm.dimshuffle(1,0,'x'), -1)],
                        outputs_info=[init, theano.shared(value=np.zeros((self.batch_size,self.n_hidden),
                                                                  dtype=theano.config.floatX),borrow=True)])
            return [self.h_l[0][:,-1,:], self.h_l[1]]



        if input_lm == None:
            self.h_l, _ = theano.scan(self.step2,
                        sequences=input.dimshuffle(1,0,2),
                        outputs_info=init)
        else:
            self.h_l, _ = theano.scan(self.step,
                        sequences=[input.dimshuffle(1,0,2),T.addbroadcast(input_lm.dimshuffle(1,0,'x'), -1)],
                        outputs_info=init)
        self.h_l = self.h_l.dimshuffle(1,0,2)
        if return_list == True:
            return self.h_l
        if return_list_except_last == True:
            return self.h_l[:,-1,:], self.h_l[:,:-1,:]
        return self.h_l[:,-1,:]

    def step2(self,x_t, h_tm1):
        x_z = T.dot(x_t, self.W_z) + self.b_z
        x_r = T.dot(x_t, self.W_r) + self.b_r
        x_h = T.dot(x_t, self.W_h) + self.b_h
        z = self.inner_activation(x_z + T.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + T.dot(h_tm1, self.U_r))

        hh = self.activation(x_h + T.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h
    def step3(self,x_t,mask, h_tm1, gate_tm1):
        #h_tm1 =  mask * h_tm1
        x_z = T.dot(x_t, self.W_z) + self.b_z
        x_r = T.dot(x_t, self.W_r) + self.b_r
        x_h = T.dot(x_t, self.W_h) + self.b_h
        z = self.inner_activation(x_z + T.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + T.dot(h_tm1, self.U_r))

        hh = self.activation(x_h + T.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        h = mask * h + (1-mask) * h_tm1

        return [h,r]

    def step(self,x_t,mask, h_tm1):
        #h_tm1 =  mask * h_tm1
        x_z = T.dot(x_t, self.W_z) + self.b_z
        x_r = T.dot(x_t, self.W_r) + self.b_r
        x_h = T.dot(x_t, self.W_h) + self.b_h
        z = self.inner_activation(x_z + T.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + T.dot(h_tm1, self.U_r))

        hh = self.activation(x_h + T.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        h = mask * h + (1-mask) * h_tm1

        return h

class SGRU(object):
    def __init__(self, n_in, n_hidden, n_out, activation=T.tanh,inner_activation=T.nnet.sigmoid,
                 output_type='real',batch_size=200):

        self.activation = activation
        self.inner_activation = inner_activation
        self.output_type = output_type

        self.batch_size = batch_size
        self.n_hidden = n_hidden

        # recurrent weights as a shared variable
        self.U_z = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.W_z = theano.shared(glorot_uniform((n_in*2,n_hidden)),borrow=True)
        self.b_z = theano.shared(value=np.zeros((n_hidden,),dtype=theano.config.floatX),borrow=True)

        self.U_r = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.W_r = theano.shared(glorot_uniform((n_in*2,n_hidden)),borrow=True)
        self.b_r = theano.shared(value=np.zeros((n_hidden,),dtype=theano.config.floatX),borrow=True)

        self.U_h = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.W_h = theano.shared(glorot_uniform((n_in*2,n_hidden)),borrow=True)
        self.b_h = theano.shared(value=np.zeros((n_hidden,),dtype=theano.config.floatX),borrow=True)

        self.params = [self.W_z,self.W_h,self.W_r,
                       self.U_h,self.U_r,self.U_z,
                       self.b_h,self.b_r,self.b_z,]

    def __call__(self, input,input_lm=None, return_list = False, return_list_except_last = False, Init_input =None,check_gate = False):
         # activation function
        if Init_input == None:
            init = theano.shared(value=np.zeros((self.batch_size,self.n_hidden),
                                                                  dtype=theano.config.floatX),borrow=True)
        else:
            init = Init_input

        if check_gate:
            self.h_l, _ = theano.scan(self.step3,
                        sequences=[input.dimshuffle(1,0,2),T.addbroadcast(input_lm.dimshuffle(1,0,'x'), -1)],
                        outputs_info=[init, theano.shared(value=np.zeros((self.batch_size,self.n_hidden),
                                                                  dtype=theano.config.floatX),borrow=True)])
                        
            return [self.h_l[0][:,-1,:], self.h_l[1]]

        if input_lm == None:
            self.h_l, _ = theano.scan(self.step2,
                        sequences=input.dimshuffle(1,0,2),
                        outputs_info=init)
        else:
            self.h_l, _ = theano.scan(self.step,
                        sequences=[input.dimshuffle(1,0,2),T.addbroadcast(input_lm.dimshuffle(1,0,'x'), -1)],
                        outputs_info=init,)
        self.h_l = self.h_l.dimshuffle(1,0,2)
        if return_list == True:
            return self.h_l
        if return_list_except_last == True:
            return self.h_l[:,-1,:], self.h_l[:,:-1,:]
        return self.h_l[:,-1,:]

    def step2(self,x_t, h_tm1):
        x_z = T.dot(x_t, self.W_z) + self.b_z
        x_r = T.dot(x_t, self.W_r) + self.b_r
        x_h = T.dot(x_t, self.W_h) + self.b_h
        z = self.inner_activation(x_z + T.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + T.dot(h_tm1, self.U_r))

        hh = self.activation(x_h + T.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h

    def step3(self,x_t,mask, h_tm1, gate_tm1):
        #h_tm1 =  mask * h_tm1
        x_z = T.dot(x_t, self.W_z) + self.b_z
        x_r = T.dot(x_t, self.W_r) + self.b_r
        x_h = T.dot(x_t, self.W_h) + self.b_h
        z = self.inner_activation(x_z + T.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + T.dot(h_tm1, self.U_r))

        hh = self.activation(x_h + T.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        h = mask * h + (1-mask) * h_tm1

        return [h,r]

    def step(self,x_t,mask, h_tm1):
        x_z = T.dot(x_t, self.W_z) + self.b_z
        x_r = T.dot(x_t, self.W_r) + self.b_r
        x_h = T.dot(x_t, self.W_h) + self.b_h
        z = self.inner_activation(x_z + T.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + T.dot(h_tm1, self.U_r))

        hh = self.activation(x_h + T.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        h = mask * h + (1-mask) * h_tm1

        return h

class WordVecs(object):
    def __init__(self, fname, vocab, binary, gensim):
        if gensim:
            word_vecs = self.load_gensim(fname,vocab)
        self.k = len(word_vecs.values()[0])
        self.W, self.word_idx_map = self.get_W(word_vecs, k=self.k)

    def get_W(self, word_vecs, k=200):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, k))
        W[0] = np.zeros(k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def load_gensim(self, fname, vocab):

         fp = open(fname)
         info = fp.readline().split()
         model = {}
         embed_dim = int(info[1])
         for line in fp:
            line = line.split()
            model[line[0]] = np.array(map(float, line[1:]), dtype='float32')
         fp.close()

         # model = Word2Vec.load(fname)
         weights = [[0.] *embed_dim]
         word_vecs = {}
         total_inside_new_embed = 0
         miss= 0
         for pair in vocab:
             word = pair.encode('utf-8')
             if word in model:
                # print(word)
                total_inside_new_embed += 1
                word_vecs[pair] = np.array([w for w in model[word]])
                #weights.append([w for w in model[word]])
             else:
                miss = miss + 1
                word_vecs[pair] = np.array([0.] * embed_dim)
                #weights.append([0.] * model.vector_size)
         print('transfer', total_inside_new_embed, 'words from the embedding file, total', len(vocab), 'candidate')
         print('miss word2vec', miss)
         return word_vecs

class LogisticRegression(object):
    def __init__(self,input,n_in,n_out,rng):
        self.W = theano.shared( np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            ))
        self.b = theano.shared(value=np.zeros(n_out,dtype=theano.config.floatX),borrow=True,name='b')
        self.predict_prob = T.nnet.softmax(T.dot(input,self.W)+self.b)
        self.predict_y = T.argmax(self.predict_prob,axis=1)
        self.params=[self.W,self.b]

    def negative_log_likelihood(self, y):
        #return - T.mean(y * T.log(self.predict_prob) + (1 - y) * T.log(1 - self.predict_prob))
        return -T.mean(T.log(self.predict_prob)[T.arange(y.shape[0]), y])

    def errors(self,y):
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.predict_y,y))
        else:
            raise NotImplementedError

class Adam(object):
     def Adam(self,cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        updates = []
        grads = T.grad(cost, params)
        i = theano.shared(as_floatX(0.))
        i_t = i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
            g_t = m_t / (T.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates

class ConvSim(object):
    def __init__(self, rng,  n_in, n_out, W=None, b=None, session_size=50, \
                 activation=T.tanh,hidden_size=100, batch_size=200):
        self.W = theano.shared(value=ortho_weight(hidden_size), borrow=True)
        self.activation = activation

        self.conv_layer = LeNetConvPoolLayer2(rng,filter_shape=(8,2,3,3),
                                    image_shape=(batch_size,2,session_size,\
                                                 session_size)
                       ,poolsize=(3,3),non_linear='relu')

        self.hidden_layer = HiddenLayer2(rng,2048,n_out)
        self.params = [self.W,] + self.conv_layer.params + self.hidden_layer.params
    def Get_M2(self,input_l,input_r):
        return T.batched_dot(T.dot(input_l,self.W),input_r.dimshuffle(0,2,1))

    def __call__(self, origin_l,origin_r,input_l,input_r):
        channel_1 = T.batched_dot(origin_l,origin_r.dimshuffle(0,2,1))
        channel_2 = T.batched_dot(T.dot(input_l,self.W),input_r.dimshuffle(0,2,1))
        input = T.stack([channel_1,channel_2],axis=1)
        mlp_in = T.flatten(self.conv_layer(input),2)

        return self.hidden_layer(mlp_in)

class HiddenLayer2(object):
    def __init__(self, rng,  n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.activation = activation

        self.params = [self.W, self.b]

    def __call__(self, input):
        lin_output = T.dot(input, self.W) + self.b
        return self.activation(lin_output)

class LeNetConvPoolLayer2(object):
    """
    Pool Layer of a convolutional network 
    """

    def __init__(self, rng, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        print('image shape', image_shape)
        print('filter shape', filter_shape)
        assert image_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /np.prod(poolsize))
        # initialize weights with random weights
        if self.non_linear=="none" or self.non_linear=="relu":
            self.W = theano.shared(np.asarray(rng.uniform(low=-0.01,high=0.01,size=filter_shape),
                                                dtype=theano.config.floatX),borrow=True,name="W_conv")
        else:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),borrow=True,name="W_conv")
        b_values =np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        self.params = [self.W, self.b]
        # convolve input feature maps with filters


    def __call__(self, input):
        conv_out = conv.conv2d(input=input, filters=self.W,filter_shape=self.filter_shape, image_shape=self.image_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True,mode="max")
        elif self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output =theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True,mode="max")
        else:
            pooled_out = theano.tensor.signal.pool.pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True,mode="max")
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.output


    def predict(self, new_data, batch_size):
        """
        predict for new data
        """
        img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=img_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        if self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = theano.tensor.signal.pool.pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output

class self_attention():
    def __init__(self, n_in):
        self.W_p = theano.shared(ortho_weight(n_in), borrow=True)
        self.W_phat = theano.shared(ortho_weight(n_in), borrow=True)
        self.U_s = theano.shared(glorot_uniform((n_in,1)), borrow=True)
        self.params = [self.W_p,self.U_s,self.W_phat]

    def __call__(self, input, input_lm):
        self.s_a, _ = theano.scan(self.self_attention, \
                                  sequences=input.dimshuffle(1,0,2), \
                                  outputs_info=None, \
                                  non_sequences=[input.dimshuffle(1,0,2), \
                                              input_lm.dimshuffle(1,0)])
        self.s_a = self.s_a.dimshuffle(1,0,2)
        return self.s_a

    def self_attention(self, x_t, x_all, x_mask_all):
        final = T.dot(T.tanh(T.dot(x_all,self.W_p) + T.dot(x_t,self.W_phat)),self.U_s)
        weight = (T.exp(T.max(final,2)) * x_mask_all).dimshuffle(1,0)
        weight2 = weight / T.sum(weight,1)[:,None]
        final2 = T.sum(x_all.dimshuffle(1,0,2)*weight[:,:,None],1)
        return final2