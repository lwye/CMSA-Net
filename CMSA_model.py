import numpy as np
import tensorflow as tf
import sys
from deeplab_resnet import model as deeplab101

from util import data_reader
from util.processing_tools import *
from util import im_processing, text_processing, eval_tools
from util import loss

class CMSA_model(object):

    def __init__(self,  batch_size = 1, 
                        num_steps = 20,
                        vf_h = 40,
                        vf_w = 40,
                        H = 320,
                        W = 320,
                        vf_dim = 2048,
                        vocab_size = 12112,
                        w_emb_dim = 1000,
                        v_emb_dim = 1000,
                        mlp_dim = 500,
                        start_lr = 0.00025,
                        lr_decay_step = 800000,
                        lr_decay_rate = 1.0,
                        rnn_size = 1000,
                        keep_prob_rnn = 1.0,
                        keep_prob_emb = 1.0,
                        keep_prob_mlp = 1.0,
                        num_rnn_layers = 1,
                        optimizer = 'adam',
                        weight_decay = 0.0005,
                        mode = 'eval',
                        weights = 'deeplab',
                        conv5 = False):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.vf_h = vf_h
        self.vf_w = vf_w
        self.H = H
        self.W = W
        self.vf_dim = vf_dim
        self.start_lr = start_lr
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.vocab_size = vocab_size
        self.w_emb_dim = w_emb_dim
        self.v_emb_dim = v_emb_dim
        self.mlp_dim = mlp_dim
        self.rnn_size = rnn_size
        self.keep_prob_rnn = keep_prob_rnn
        self.keep_prob_emb = keep_prob_emb
        self.keep_prob_mlp = keep_prob_mlp
        self.num_rnn_layers = num_rnn_layers
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.mode = mode
        self.weights = weights
        self.conv5 = conv5

        self.words = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.im = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 3])
        self.target_fine = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 1])

            
        resmodel = deeplab101.DeepLabResNetModel({'data': self.im}, is_training=False)
        self.visual_feat = resmodel.layers['res5c_relu']
            
        self.visual_feat_c4 = resmodel.layers['res4b22_relu']
        self.visual_feat_c3 = resmodel.layers['res3b3_relu']

        with tf.variable_scope("text_objseg"):
            self.build_graph()
            if self.mode == 'eval':
                return
            self.train_op()

    def build_graph(self):

        if self.weights == 'deeplab':
            # atrous0 = self._atrous_conv("atrous0", self.visual_feat, 3, self.vf_dim, self.v_emb_dim, 6)
            # atrous1 = self._atrous_conv("atrous1", self.visual_feat, 3, self.vf_dim, self.v_emb_dim, 12)
            # atrous2 = self._atrous_conv("atrous2", self.visual_feat, 3, self.vf_dim, self.v_emb_dim, 18)
            # atrous3 = self._atrous_conv("atrous3", self.visual_feat, 3, self.vf_dim, self.v_emb_dim, 24)
            # visual_feat = tf.add(atrous0, atrous1)
            # visual_feat = tf.add(visual_feat, atrous2)
            # visual_feat = tf.add(visual_feat, atrous3)
            visual_feat_c5 = self._conv("mlp_c5", self.visual_feat, 1, self.vf_dim, self.v_emb_dim, [1, 1, 1, 1])
            
        embedding_mat = tf.get_variable("embedding", [self.vocab_size, self.w_emb_dim], 
                                        initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
        embedded_seq = tf.nn.embedding_lookup(embedding_mat, tf.transpose(self.words))


        # Generate spatial feature
        spatial = tf.convert_to_tensor(generate_spatial_batch(self.batch_size, self.vf_h, self.vf_w))
        visual_feat_c5 = tf.nn.l2_normalize(visual_feat_c5, 3)
        
        visual_feat_c4 = tf.nn.l2_normalize(self.visual_feat_c4, 3)
        visual_feat_c3 = tf.nn.l2_normalize(self.visual_feat_c3, 3)
        
        
        def f1():
            return tf.constant(0., shape=[1,40,40, 2008]), tf.constant(0., shape=[1,40,40, 1024+1000+8]),tf.constant(0., shape = [1,40,40,512+1000+8])

        def f2():
            w_emb = embedded_seq[n, :, :]
            
            lang_feat = tf.reshape(w_emb, [self.batch_size, 1, 1, self.rnn_size])
            lang_feat = tf.nn.l2_normalize(lang_feat, 3)
            lang_feat = tf.tile(lang_feat, [1, self.vf_h, self.vf_w, 1])

            feat_all_c5 = tf.concat([visual_feat_c5, lang_feat, spatial], 3) # batch h w c
            feat_all_c4 = tf.concat([visual_feat_c4, lang_feat, spatial], 3) # batch h w c
            feat_all_c3 = tf.concat([visual_feat_c3, lang_feat, spatial], 3) # batch h w c
            return feat_all_c5, feat_all_c4, feat_all_c3
    
        
        feat_c5_list=[]
        feat_c4_list=[]
        feat_c3_list=[]
        with tf.variable_scope("RNN"):
            for n in range(self.num_steps): # num_words
                if n > 0:
                    tf.get_variable_scope().reuse_variables()

                feat_c5, feat_c4, feat_c3  = tf.cond(tf.equal(self.words[0, n], tf.constant(0)), f1, f2)
                feat_c5_list.append(feat_c5)
                feat_c4_list.append(feat_c4)
                feat_c3_list.append(feat_c3)


        word_where = tf.transpose(tf.not_equal(self.words, tf.constant(0)))
        self.feat_c5 = tf.boolean_mask(feat_c5_list, word_where) 
        self.feat_c5 = tf.expand_dims(self.feat_c5, 0) # batch, n, h, w, c
        self.feat_c4 = tf.boolean_mask(feat_c4_list, word_where) 
        self.feat_c4 = tf.expand_dims(self.feat_c4, 0) # batch, n, h, w, c
        self.feat_c3 = tf.boolean_mask(feat_c3_list, word_where) 
        self.feat_c3 = tf.expand_dims(self.feat_c3, 0) # batch, n, h, w, c
        
        #cross-modal self-attention 
        self.feat_c5 = self.cmsa_layer(self.feat_c5, 'CMSA', dim = 512, sub =2, out_dim = 2008)
        
        c5_output = tf.layers.conv2d(self.feat_c5, filters= 500, kernel_size= 1, padding='SAME', dilation_rate=(1, 1),  activation= tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer())
        
        
        self.feat_c4 = tf.layers.conv3d(self.feat_c4, filters= 256, kernel_size= 1, padding='SAME', dilation_rate=(1, 1, 1),  activation= tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer())
        c4_output = self.cmsa_layer(self.feat_c4, 'CMSA_C4', dim = 128, sub =2, out_dim = 256)
        c4_output = tf.layers.conv2d(c4_output, filters= 500, kernel_size= 1, padding='SAME', dilation_rate=(1, 1),  activation= tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer())

        
        self.feat_c3 = tf.layers.conv3d(self.feat_c3, filters= 128, kernel_size= 1, padding='SAME', dilation_rate=(1, 1, 1),  activation= tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer())
        c3_output = self.cmsa_layer(self.feat_c3, 'CMSA_C3', dim = 64, sub =2, out_dim = 128)
        c3_output = tf.layers.conv2d(c3_output, filters= 500, kernel_size= 1, padding='SAME', dilation_rate=(1, 1),  activation= tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer())
        
        #Gated Multi-Level Fusion
        feats_out = self.MGATE('mgate', c5_output,  c4_output, c3_output, c_dim = 500) # 
        score = self._conv("score", feats_out, 3, self.mlp_dim, 1, [1, 1, 1, 1])

        self.pred = score
        self.up = tf.image.resize_bilinear(self.pred, [self.H, self.W])
        self.sigm = tf.sigmoid(self.up)

    
    def MGATE(self, name, h_feats, l1_feats, l2_feats, c_dim, alpha = 0.5 ):
        with tf.variable_scope(name):
            x1 , x2 , x3 = h_feats, l1_feats, l2_feats # batch h w c
            x1_out = self.GATECell('x1', x1, x2,x3, c_dim, alpha)
            x2_out = self.GATECell('x2', x2, x1,x3, c_dim, alpha)
            x3_out = self.GATECell('x3', x3, x1,x2, c_dim, alpha)
            out = x1_out + x2_out + x3_out
            
            return out

    def GATECell(self, name, x1, x2, x3, c_dim, alpha ):
        with tf.variable_scope(name):
            y = tf.layers.conv2d(x1, filters= c_dim*3, kernel_size= 3, padding='SAME', dilation_rate=(1, 1),  activation= None, kernel_initializer= tf.contrib.layers.xavier_initializer())
            i, f, r = tf.split(y, 3, axis= 3)
            f = tf.sigmoid(f + 1.0) 
            r = tf.sigmoid(r + 1.0) 
            a = tf.Variable(alpha, trainable=True)
            c = a*f*x2 + (1-a)*f*(x3) + (1-f)*i
            out = r * tf.tanh(c) + (1-r)*x1
            return out
    
    def cmsa_layer(self, in_feats, name, dim= 512, sub = 2, out_dim = 2008):
        with tf.variable_scope(name):
            theta = tf.layers.conv3d(in_feats, filters= dim, kernel_size= 1, padding='SAME', dilation_rate=(1, 1, 1),  activation= tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer())
            theta = tf.reshape(theta, [self.batch_size,-1, dim])
            phi = tf.layers.conv3d(in_feats, filters= dim, kernel_size= 1, padding='SAME', dilation_rate=(1, 1, 1),  activation= tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer())
            phi = tf.layers.max_pooling3d(phi, pool_size = sub, strides=sub, padding='same')
            phi = tf.reshape(phi, [self.batch_size,-1, dim])
            phi = tf.transpose(phi, perm=[0,2,1])
            feat_nl = tf.matmul(theta, phi) # b, thw, thw
            feat_nl = tf.nn.softmax(feat_nl, -1)
   
            feats = tf.layers.conv3d(in_feats, filters= dim, kernel_size= 1, padding='SAME', dilation_rate=(1, 1, 1),  activation= tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer()) # bthwc
            feats = tf.layers.max_pooling3d(feats, pool_size = sub, strides=sub, padding='same')
            feats = tf.reshape(feats, [self.batch_size,-1, dim])
            feats = tf.matmul(feat_nl, feats)
            feats = tf.reshape(feats, [self.batch_size,-1, 40,40, dim])
            
            feats = tf.layers.conv3d(feats, filters= out_dim, kernel_size= 1, padding='SAME', dilation_rate=(1, 1, 1),  activation= tf.nn.relu, kernel_initializer= tf.contrib.layers.xavier_initializer())
            feats = feats + in_feats 
            
            feats = tf.reduce_mean(feats, axis=1, keep_dims=False)
            return feats

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], 
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.conv2d(x, w, strides, padding='SAME') + b


    def train_op(self):
        if self.conv5:
            tvars = [var for var in tf.trainable_variables() if var.op.name.startswith('text_objseg') 
                        or var.name.startswith('res5') or var.name.startswith('res4')
                        or var.name.startswith('res3')]
        else:
            tvars = [var for var in tf.trainable_variables() if var.op.name.startswith('text_objseg')]
        reg_var_list = [var for var in tvars if var.op.name.find(r'DW') > 0 or var.name[-9:-2] == 'weights']
        print('Collecting variables for regularization:')
        for var in reg_var_list: print('\t%s' % var.name)
        print('Done.')

        # define loss
        self.target = tf.image.resize_bilinear(self.target_fine, [self.vf_h, self.vf_w])
        self.cls_loss = loss.weighed_logistic_loss(self.up, self.target_fine, 1, 1)
        self.reg_loss = loss.l2_regularization_loss(reg_var_list, self.weight_decay)
        self.cost = self.cls_loss + self.reg_loss

        # learning rate
        lr = tf.Variable(0.0, trainable=False)
        self.learning_rate = tf.train.polynomial_decay(self.start_lr, lr, self.lr_decay_step, end_learning_rate=0.00001, power=0.9)

        # optimizer
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError("Unknown optimizer type %s!" % self.optimizer)

        # learning rate multiplier
        grads_and_vars = optimizer.compute_gradients(self.cost, var_list=tvars)
        # var_lr_mult = {var: (2.0 if var.op.name.find(r'biases') > 0 else 1.0) for var in tvars}
        var_lr_mult = {}
        for var in tvars:
            if var.op.name.find(r'biases') > 0:
                var_lr_mult[var] = 2.0
            elif var.name.startswith('res5') or var.name.startswith('res4') or var.name.startswith('res3'):
                var_lr_mult[var] = 1.0
            else:
                var_lr_mult[var] = 1.0
        print('Variable learning rate multiplication:')
        for var in tvars:
            print('\t%s: %f' % (var.name, var_lr_mult[var]))
        print('Done.')
        grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v) for g, v in grads_and_vars]
        # grads_and_vars = [((g if not v.name.startswith('res5') else tf.clip_by_norm(g, 0.1)), v) for g, v in grads_and_vars]

        # training step
        self.train_step = optimizer.apply_gradients(grads_and_vars, global_step=lr)
