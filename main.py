from __future__ import print_function

import cPickle
import numpy as np
import theano
from gensim.models.word2vec import Word2Vec
import theano.tensor as T
import argparse
from model import GRU, SGRU, WordVecs, LogisticRegression, Adam, ConvSim, self_attention

parser = argparse.ArgumentParser(description="The param of the training")
parser.add_argument('--dataset', type=str, \
                    default='../data/douban.train.small.done', \
                    help='The location of the train dataset')
parser.add_argument('--save_result', type=str, \
                    default='../result/result.txt', \
                    help='The location of the pred result')
parser.add_argument('--n_epoch', type=int, default=6, \
                    help='The number of epoches of training')
parser.add_argument('--batch_size', type=int, default=170, \
                    help='The batch size of the each training time')
parser.add_argument('--max_length', type=int, default=50, \
                    help='The max length of a sentence')
parser.add_argument('--hidden_size', type=int, default=200, \
                    help='The hidden size of the first RNN')
parser.add_argument('--word_embedding_size', type=int, default=200, \
                    help='The input size of the first RNN')
parser.add_argument('--session_hidden_size', type=int, default=50, \
                    help='The hidden size of the second RNN')
parser.add_argument('--session_input_size', type=int, default=50, \
                    help='The input size of the second RNN')
parser.add_argument('--model_name', type=str, default='DUA.bin', \
                    help='The name of the model file')
parser.add_argument('--val_frequency', type=int, default=100, \
                    help='The frequency to valid the model')
parser.add_argument('--max_turn', type=int, default=10, \
                    help='The number of the multiturn conversation')
parser.add_argument('--learning_rate', type=float, default=0.005, \
                    help='The learning rate of the training')
parser.add_argument('--r_seed', type=int, default=0, \
                    help='The random seed of the data')
parser.add_argument('--ismask', type=bool, default=True, \
                    help='The flag to indicate whether to use mask')
args = parser.parse_args()

max_turn = args.max_turn
sf = open(args.save_result, 'w')

def get_idx_from_sent_msg(sents, word_idx_map, max_l=50, mask=True):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    turns = []
    for sent in sents.split('_t_'):
        x = [0] * max_l
        x_mask = [0.] * max_l
        words = sent.split()
        length = len(words)
        for i, word in enumerate(words):
            if max_l - length + i < 0: continue
            if word in word_idx_map:
                x[max_l - length + i] = word_idx_map[word]
            #if x[max_l - length + i] != 0:
            x_mask[max_l - length + i] = 1
        if mask:
            x += x_mask
        turns.append(x)

    final = [0.] * (max_l * 2 * max_turn)
    for i in range(max_turn):
        if max_turn - i <= len(turns):
            for j in range(max_l * 2):
                final[i*(max_l*2) + j] = turns[-(max_turn-i)][j]
    return final

def get_idx_from_sent(sent, word_idx_map, max_l=50, mask=True):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = [0] * max_l
    x_mask = [0.] * max_l
    words = sent.split()
    length = len(words)
    for i, word in enumerate(words):
        if max_l - length + i < 0: continue
        if word in word_idx_map:
            x[max_l - length + i] = word_idx_map[word]
        #if x[max_l - length + i] != 0:
        x_mask[max_l - length + i] = 1
    if mask:
        x += x_mask
    return x

def get_session_mask(sents):
    session_mask = [0.] * max_turn
    turns = []
    for sent in sents.split('_t_'):
        words = sent.split()
        if len(words) > 0:
            turns.append(len(words))

    for i in range(max_turn):
        if max_turn - i <= len(turns):
            session_mask[-(max_turn-i)] = 1.
    return session_mask

def _dropout_from_layer(rng, layer, p):
    """
    p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')

def load_params(params, filename):
    f = open(filename)
    num_params = cPickle.load(f)
    for p, w in zip(params, num_params):
        p.set_value(w.astype('float32'), borrow=True)
    sf.write("load successfully")
    sf.write('\n')
    sf.flush()

def glorot_uniform(size):
    fan_in, fan_out = size
    s = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(size=size, low=-s, high=s).astype(theano.config.floatX)

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

    def unpack(self, input, input_lm):
        self.weight2, _ = theano.scan(self.self_att, \
                                  sequences=input.dimshuffle(1,0,2), \
                                  outputs_info=None, \
                                  non_sequences=[input.dimshuffle(1,0,2), \
                                              input_lm.dimshuffle(1,0)])
        return self.weight2

    def self_attention(self, x_t, x_all, x_mask_all):
        final = T.dot(T.tanh(T.dot(x_all,self.W_p) + T.dot(x_t,self.W_phat)),self.U_s)
        weight = (T.exp(T.max(final,2)) * x_mask_all).dimshuffle(1,0)
        weight2 = weight / T.sum(weight,1)[:,None]
        final2 = T.sum(x_all.dimshuffle(1,0,2)*weight[:,:,None],1)
        return final2
    
    def self_att(self, x_t, x_all, x_mask_all):
        final = T.dot(T.tanh(T.dot(x_all,self.W_p) + T.dot(x_t,self.W_phat)),self.U_s)
        weight = (T.exp(T.max(final,2)) * x_mask_all).dimshuffle(1,0)
        weight2 = weight / T.sum(weight,1)[:,None]
        final2 = T.sum(x_all.dimshuffle(1,0,2)*weight[:,:,None],1)
        return weight2

def main(datasets, U, n_epochs=20, batch_size=20, max_l=100, hidden_size=100, \
         word_embedding_size=100, session_hidden_size=50, session_input_size=50, \
         model_name='SMN_last.bin', learning_rate=0.001, r_seed=3435, \
        val_frequency=100):
    hiddensize = hidden_size
    U = U.astype(dtype=theano.config.floatX)
    rng = np.random.RandomState(r_seed)
    lsize, rsize = max_l, max_l
    sessionmask = T.matrix()
    lx = []
    lxmask = []
    for i in range(max_turn):
        lx.append(T.matrix())
        lxmask.append(T.matrix())

    index = T.lscalar()
    rx = T.matrix('rx')
    rxmask = T.matrix()
    y = T.ivector('y')
    Words = theano.shared(value=U, name="Words")
    llayer0_input = []
    for i in range(max_turn):
        llayer0_input.append(Words[T.cast(lx[i].flatten(), dtype="int32")] \
            .reshape((lx[i].shape[0], lx[i].shape[1], Words.shape[1])))

    # input: word embeddings of the mini batch
    rlayer0_input = Words[T.cast(rx.flatten(), dtype="int32")].\
                    reshape((rx.shape[0], rx.shape[1], Words.shape[1]))

    train_set, dev_set, test_set = datasets[0], datasets[1], datasets[2]

    train_set_lx = []
    train_set_lx_mask = []
    q_embedding = []
    q_embedding_Cat = []
    q_embedding_Cat_mask = []
    q_embedding_self_att = []
    q_embedding_self_att_rnn = []
    q_embedding_hiddenequal = []

    offset = 2 * lsize
    for i in range(max_turn):
        train_set_lx.append(theano.shared(
            np.asarray(a=train_set[:, offset*i:offset*i+lsize], \
                       dtype=theano.config.floatX), \
                       borrow=True))
        train_set_lx_mask.append(theano.shared(
            np.asarray(a=train_set[:, offset*i + lsize:offset*i + 2*lsize], \
                       dtype=theano.config.floatX), \
                       borrow=True))
    train_set_rx = theano.shared(
        np.asarray(a=train_set[:, offset*max_turn:offset*max_turn + lsize], \
                   dtype=theano.config.floatX), \
                   borrow=True)
    train_set_rx_mask = theano.shared(
        np.asarray(a=train_set[:, offset*max_turn+lsize:offset*max_turn + 2*lsize], \
                   dtype=theano.config.floatX), \
                   borrow=True)
    train_set_session_mask = theano.shared(
        np.asarray(a=train_set[:, -max_turn-1:-1], \
                   dtype=theano.config.floatX), \
                   borrow=True)
    train_set_y = theano.shared(np.asarray(train_set[:, -1], dtype="int32"), \
                               borrow=True)

    val_set_lx = []
    val_set_lx_mask = []
    for i in range(max_turn):
        val_set_lx.append(theano.shared(
            np.asarray(a=dev_set[:, offset*i:offset*i + lsize], \
                       dtype=theano.config.floatX), \
                       borrow=True))
        val_set_lx_mask.append(theano.shared(
            np.asarray(a=dev_set[:, offset*i + lsize:offset*i + 2*lsize], \
                       dtype=theano.config.floatX), \
                       borrow=True))
    val_set_rx = theano.shared(
        np.asarray(a=dev_set[:, offset*max_turn:offset*max_turn + lsize], \
                   dtype=theano.config.floatX), \
                   borrow=True)
    val_set_rx_mask = theano.shared(
        np.asarray(a=dev_set[:, offset*max_turn + lsize:offset*max_turn + 2*lsize], \
                   dtype=theano.config.floatX), \
                   borrow=True)
    val_set_session_mask = theano.shared(np.asarray(a=dev_set[:, -max_turn-1:-1], \
                                                    dtype=theano.config.floatX), \
                                         borrow=True)
    val_set_y = theano.shared(np.asarray(dev_set[:, -1], dtype="int32"), borrow=True)

    test_set_lx = []
    test_set_lx_mask = []
    for i in range(max_turn):
        test_set_lx.append(theano.shared(
            np.asarray(a=test_set[:, offset*i:offset*i + lsize], \
                       dtype=theano.config.floatX), \
                       borrow=True))
        test_set_lx_mask.append(theano.shared(
            np.asarray(a=test_set[:, offset*i + lsize:offset*i + 2*lsize], \
                       dtype=theano.config.floatX), \
                       borrow=True))
    test_set_rx = theano.shared(
        np.asarray(a=test_set[:, offset*max_turn:offset*max_turn + lsize], \
                   dtype=theano.config.floatX), \
                   borrow=True)
    test_set_rx_mask = theano.shared(
        np.asarray(a=test_set[:, offset*max_turn + lsize:offset*max_turn + 2*lsize], \
                   dtype=theano.config.floatX), \
                   borrow=True)
    test_set_session_mask = theano.shared(np.asarray(a=test_set[:, -max_turn-1:-1], \
                                                    dtype=theano.config.floatX), \
                                         borrow=True)
    test_set_y = theano.shared(np.asarray(test_set[:, -1], dtype="int32"), \
                               borrow=True)

    dic = {}
    for i in range(max_turn):
        dic[lx[i]] = train_set_lx[i][index*batch_size:(index+1)*batch_size]
        dic[lxmask[i]] = train_set_lx_mask[i][index*batch_size:(index+1)*batch_size]
    dic[rx] = train_set_rx[index*batch_size:(index+1)*batch_size]
    dic[sessionmask] = train_set_session_mask[index*batch_size:(index+1)*batch_size]
    dic[rxmask] = train_set_rx_mask[index*batch_size:(index+1)*batch_size]
    dic[y] = train_set_y[index*batch_size:(index+1)*batch_size]

    val_dic = {}
    for i in range(max_turn):
        val_dic[lx[i]] = val_set_lx[i][index*batch_size:(index+1)*batch_size]
        val_dic[lxmask[i]] = val_set_lx_mask[i][index*batch_size:(index+1)*batch_size]
    val_dic[rx] = val_set_rx[index*batch_size:(index+1)*batch_size]
    val_dic[sessionmask] = val_set_session_mask[index*batch_size:(index+1)*batch_size]
    val_dic[rxmask] = val_set_rx_mask[index*batch_size:(index+1)*batch_size]
    val_dic[y] = val_set_y[index*batch_size:(index+1)*batch_size]

    test_dic = {}
    for i in range(max_turn):
        test_dic[lx[i]] = test_set_lx[i][index*batch_size:(index+1)*batch_size]
        test_dic[lxmask[i]] = test_set_lx_mask[i][index*batch_size:(index+1)*batch_size]
    test_dic[rx] = test_set_rx[index*batch_size:(index+1)*batch_size]
    test_dic[sessionmask] = test_set_session_mask[index*batch_size:(index+1)*batch_size]
    test_dic[rxmask] = test_set_rx_mask[index*batch_size:(index+1)*batch_size]
    test_dic[y] = test_set_y[index*batch_size:(index+1)*batch_size]

    # This is the first RNN.
    sentence2vec = GRU(n_in=word_embedding_size, n_hidden=hiddensize, \
                       n_out=hiddensize, batch_size=batch_size)
    for i in range(max_turn):
        q_embedding.append(sentence2vec(llayer0_input[i], lxmask[i], True))
    r_embedding = sentence2vec(rlayer0_input, rxmask, True)

    # This is the concat/elementwise_produce of the after the first RNN which 
    # concat the tenth sentence to the first nine sentences.
    for i in range(max_turn):
        q_embedding_Cat.append(T.concatenate([q_embedding[i], \
                                              q_embedding[-1]], \
                               axis=2))
        q_embedding_Cat_mask.append(lxmask[i])
    r_embedding_Cat = T.concatenate([r_embedding, q_embedding[-1]], axis=2)
    r_embedding_Cat_mask = rxmask
    # This is the self_attention step
    sa = self_attention(n_in=hiddensize*2)
    for i in range(max_turn):
        q_embedding_self_att.append(T.concatenate([q_embedding_Cat[i], \
                                                   sa(q_embedding_Cat[i], \
                                                       q_embedding_Cat_mask[i])], \
                                                  axis=2))
    r_embedding_self_att = T.concatenate([r_embedding_Cat, \
                                          sa(r_embedding_Cat, \
                                              r_embedding_Cat_mask)], \
                                         axis=2)
    # This is the SRNN
    vec2svec = SGRU(n_in=hiddensize*2, n_hidden=hiddensize, \
                    n_out=hiddensize, batch_size=batch_size)

    for i in range(max_turn):
        q_embedding_self_att_rnn.append(vec2svec(q_embedding_self_att[i], \
                                                 q_embedding_Cat_mask[i], \
                                                 True))
    r_embedding_self_att_rnn = vec2svec(r_embedding_self_att, \
                                        r_embedding_Cat_mask, \
                                        True)

    # This is the CNN with pooling and full-connection
    pooling_layer = ConvSim(rng=rng, n_in=max_l, n_out=session_input_size, \
                            hidden_size=hiddensize, session_size=session_hidden_size, \
                            batch_size=batch_size)
    poolingoutput = []
    for i in range(max_turn):
        poolingoutput.append(pooling_layer(llayer0_input[i], \
                                           rlayer0_input, \
                                           q_embedding_self_att_rnn[i], \
                                           r_embedding_self_att_rnn))

    # This is the second RNN 
    session2vec = GRU(n_in=session_input_size, n_hidden=session_hidden_size, \
                      n_out=session_hidden_size, batch_size=batch_size)
    res = session2vec(T.stack(poolingoutput, 1), sessionmask, True)

    # This is the final Attention and put the output to a classifier
    W = theano.shared(ortho_weight(session_hidden_size), borrow=True)
    W2 = theano.shared(glorot_uniform((hiddensize, session_hidden_size)), borrow=True)
    b = theano.shared(value=np.zeros((session_hidden_size, ), dtype='float32'), borrow=True)
    U_s = theano.shared(glorot_uniform((session_hidden_size, 1)), borrow=True)

    final = T.dot(T.tanh(T.dot(res, W) + \
                         T.dot(T.stack(q_embedding_self_att_rnn, 1)[:, :, -1, :], W2) \
                         + b), U_s)
    weight = T.exp(T.max(final, 2)) * sessionmask
    weight2 = weight / T.sum(weight, 1)[:, None]
    final2 = T.sum(res*weight2[:, :, None], 1)+1e-6

    # This is the classifier
    classifier = LogisticRegression(final2, session_hidden_size, 2, rng)

    # Calculate the cost and updata the param with gradient
    cost = classifier.negative_log_likelihood(y)
    error = classifier.errors(y)
    predict = classifier.predict_prob
    opt = Adam()

    # Make params
    params = classifier.params
    params += sentence2vec.params
    params += session2vec.params
    params += pooling_layer.params
    params += [Words, W, b, W2, U_s]
    params += vec2svec.params
    params += sa.params

    # Make updater
    grad_updates = opt.Adam(cost=cost, params=params, lr=learning_rate)

    # The training step
    train_model = theano.function([index], cost, updates=grad_updates, \
                                  givens=dic, on_unused_input='ignore')
    val_model = theano.function([index], [cost, error], givens=val_dic, \
                                on_unused_input='ignore')
    best_dev = 1.
    n_train_batches = datasets[0].shape[0]/batch_size
    for i in xrange(n_epochs):
        cost_all = 0
        total = 0.
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            batch_cost = train_model(minibatch_index)
            total = total + 1
            cost_all = cost_all + batch_cost
            if total % val_frequency == 0:
                sf.write('epcho %d, num %d, train_loss %f' %(i, total, cost_all/total))
                sf.write('\n')
                sf.flush()
                cost_dev = 0
                errors_dev = 0
                j = 0
                for minibatch_index in xrange(datasets[1].shape[0]/batch_size):
                    tcost, terr = val_model(minibatch_index)
                    cost_dev += tcost
                    errors_dev += terr
                    j = j+1
                cost_dev = cost_dev / j
                errors_dev = errors_dev / j
                if cost_dev < best_dev:
                    best_dev = cost_dev
                    save_params(params, model_name+'dev')
                sf.write("epcho %d, num %d, dev_loss %f" % (i, total, cost_dev))
                sf.write('\n')
                sf.write("epcho %d, num %d, dev_accuracy %f" % (i, total, 1-errors_dev))
                sf.write('\n')
                sf.flush()
        cost_all = cost_all / n_train_batches
        sf.write("epcho %d loss %f" % (i, cost_all))
        sf.write('\n')
        sf.flush()

def save_params(params,filename):
    num_params = [p.get_value() for p in params]
    f = open(filename,'wb')
    cPickle.dump(num_params,f)

def make_data(revs, word_idx_map, max_l=50, ismask=True):
    """
    Transforms sentences into a 2-d matrix.
    """
    data = []
    for rev in revs:
        sent = get_idx_from_sent_msg(rev["m"], word_idx_map, max_l, ismask)
        sent += get_idx_from_sent(rev["r"], word_idx_map, max_l, ismask)
        sent += get_session_mask(rev["m"])
        sent.append(int(rev["y"]))
        data.append(sent)

    data = np.array(data,dtype="int")
    return data

if __name__=="__main__":
    x = cPickle.load(open(args.dataset,"rb"))
    revs, wordvecs, max_l = x[0], x[1], x[2]
    datasets = []

    for i in range(3):
        datasets.append(make_data(revs=revs[i], \
                                  word_idx_map=wordvecs.word_idx_map, \
                                  max_l=args.max_length, \
                                  ismask=args.ismask))
        sf.write('Dataset %d: %d' %(i,len(datasets[i])))
        sf.write('\n')
        sf.flush()


    main(datasets, wordvecs.W, batch_size=args.batch_size, \
         max_l=args.max_length, hidden_size=args.hidden_size, \
         word_embedding_size=args.word_embedding_size, \
         model_name=args.model_name, n_epochs=args.n_epoch, \
         session_hidden_size=args.session_hidden_size, \
         session_input_size=args.session_input_size, \
         learning_rate=args.learning_rate, r_seed=args.r_seed, \
         val_frequency=args.val_frequency)

sf.close()
