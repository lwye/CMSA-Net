from __future__ import division

import sys
import os
import argparse
#import ipdb
import numpy as np
import tensorflow as tf
import skimage

from CMSA_model import CMSA_model
from pydensecrf import densecrf

from util import data_reader
from util.processing_tools import *
from util import im_processing, text_processing, eval_tools


def train(modelname, max_iter, snapshot, dataset, weights, setname, mu, lr, bs, tfmodel_folder, conv5, re_iter):

    iters_per_log = 50000
    data_folder = './' + dataset + '/' + setname + '_batch/'
    data_prefix = dataset + '_' + setname
    
    tfmodel_folder = './' + dataset + '/tfmodel/CMSA/'
    snapshot_file = os.path.join(tfmodel_folder, dataset + '_' + weights + '_' + modelname + '_iter_%d.tfmodel')
    
    
    if not os.path.isdir(tfmodel_folder):
        os.makedirs(tfmodel_folder)

    cls_loss_avg = 0
    avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
    decay = 0.99
    vocab_size = 8803 if dataset == 'referit' else 12112
    

    model = CMSA_model(mode='train', vocab_size=vocab_size, weights=weights, start_lr=lr, batch_size=bs, conv5=conv5)

    if re_iter is None:
        pretrained_model = 'models/deeplab_resnet_init.ckpt'
            #pretrained_model = 'models/deeplab_resnet.ckpt'
        load_var = {var.op.name: var for var in tf.global_variables()
                    if var.name.startswith('res') or var.name.startswith('bn') or var.name.startswith('conv1')}
        snapshot_loader = tf.train.Saver(load_var)
        snapshot_saver = tf.train.Saver(max_to_keep = 1000)
        re_iter = 0
    else:
        print('resume from %d' % re_iter)
        pretrained_model = os.path.join(tfmodel_folder, dataset + '_' + weights + '_' + modelname  + '_iter_' + str(re_iter) + '.tfmodel')
        snapshot_loader = tf.train.Saver()
        snapshot_saver = tf.train.Saver(max_to_keep = 1000)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    snapshot_loader.restore(sess, pretrained_model)

    im_h, im_w, num_steps = model.H, model.W, model.num_steps
    text_batch = np.zeros((bs, num_steps), dtype=np.float32)
    image_batch = np.zeros((bs, im_h, im_w, 3), dtype=np.float32)
    mask_batch = np.zeros((bs, im_h, im_w, 1), dtype=np.float32)

    reader = data_reader.DataReader(data_folder, data_prefix)


    for n_iter in range(max_iter - re_iter):
        n_iter += re_iter    
        for n_batch in range(bs):
            batch = reader.read_batch(is_log = (n_batch==0 and n_iter%iters_per_log==0))
            text = batch['text_batch']
            im = batch['im_batch'].astype(np.float32)
            mask = np.expand_dims(batch['mask_batch'].astype(np.float32), axis=2)

            im = im[:,:,::-1]
            im -= mu

            text_batch[n_batch, ...] = text
            image_batch[n_batch, ...] = im
            mask_batch[n_batch, ...] = mask

        _, cls_loss_val, lr_val, scores_val, label_val   = sess.run([model.train_step, 
            model.cls_loss, 
            model.learning_rate, 
            model.pred, 
            model.target 
            ], 
            feed_dict={
                model.words: text_batch, 
                model.im: image_batch, 
                model.target_fine: mask_batch 
            })

        cls_loss_avg = decay*cls_loss_avg + (1-decay)*cls_loss_val

        # Accuracy
        accuracy_all, accuracy_pos, accuracy_neg = compute_accuracy(scores_val, label_val)
        avg_accuracy_all = decay*avg_accuracy_all + (1-decay)*accuracy_all
        avg_accuracy_pos = decay*avg_accuracy_pos + (1-decay)*accuracy_pos
        avg_accuracy_neg = decay*avg_accuracy_neg + (1-decay)*accuracy_neg

        if n_iter%iters_per_log==0:
            print('iter = %d, loss (cur) = %f, loss (avg) = %f, lr = %f' 
                    % (n_iter, cls_loss_val, cls_loss_avg, lr_val))
            #print('iter = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
            #        % (n_iter, accuracy_all, accuracy_pos, accuracy_neg))
            print('iter = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
                    % (n_iter, avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg))

        # Save snapshot
        if (n_iter+1) % snapshot == 0 or (n_iter+1) >= max_iter:
            snapshot_saver.save(sess, snapshot_file % (n_iter+1))
            print('snapshot saved to ' + snapshot_file % (n_iter+1))

    print('Optimization done.')


def test(modelname, iter, dataset,  weights, setname, dcrf, mu, tfmodel_folder):
    data_folder = './' + dataset + '/' + setname + '_batch/'
    data_prefix = dataset + '_' + setname
    
    tfmodel_folder = './' + dataset + '/tfmodel/CMSA'

    pretrained_model = os.path.join(tfmodel_folder, dataset + '_' +  modelname  + '_release' + '.tfmodel')
    
    score_thresh = 1e-9
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    cum_I, cum_U = 0, 0
    mean_IoU, mean_dcrf_IoU = 0, 0
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    if dcrf:
        cum_I_dcrf, cum_U_dcrf = 0, 0
        seg_correct_dcrf = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0.
    H, W = 320, 320
    vocab_size = 8803 if dataset == 'referit' else 12112
    IU_result = list()

    model = CMSA_model(H=H, W=W, mode='eval', vocab_size=vocab_size, weights=weights)

    # Load pretrained model
    snapshot_restorer = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    snapshot_restorer.restore(sess, pretrained_model)
    reader = data_reader.DataReader(data_folder, data_prefix, shuffle=False)

    NN = reader.num_batch
    print('test in', dataset, setname)
    for n_iter in range(reader.num_batch):

        if n_iter % (NN//50) == 0:
            if n_iter/(NN//50)%5 == 0:
                sys.stdout.write(str(n_iter/(NN//50)//5))
            else:
                sys.stdout.write('.')
            sys.stdout.flush()

        batch = reader.read_batch(is_log = False)
        text = batch['text_batch']
        im = batch['im_batch']
        mask = batch['mask_batch'].astype(np.float32)

        proc_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, H, W))
        proc_im_ = proc_im.astype(np.float32)
        proc_im_ = proc_im_[:,:,::-1]
        proc_im_ -= mu

        scores_val, up_val, sigm_val = sess.run([model.pred, model.up, model.sigm],
            feed_dict={
                model.words: np.expand_dims(text, axis=0),
                model.im: np.expand_dims(proc_im_, axis=0)
            })

        up_val = np.squeeze(up_val)
        pred_raw = (up_val >= score_thresh).astype(np.float32)
        predicts = im_processing.resize_and_crop(pred_raw, mask.shape[0], mask.shape[1])
        if dcrf:
            # Dense CRF post-processing
            sigm_val = np.squeeze(sigm_val)
            d = densecrf.DenseCRF2D(W, H, 2)
            U = np.expand_dims(-np.log(sigm_val), axis=0)
            U_ = np.expand_dims(-np.log(1 - sigm_val), axis=0)
            unary = np.concatenate((U_, U), axis=0)
            unary = unary.reshape((2, -1))
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=proc_im, compat=10)
            Q = d.inference(5)
            pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)
            predicts_dcrf = im_processing.resize_and_crop(pred_raw_dcrf, mask.shape[0], mask.shape[1])


        I, U = eval_tools.compute_mask_IU(predicts, mask)
        IU_result.append({'batch_no': n_iter, 'I': I, 'U': U})
        mean_IoU += float(I) / U
        cum_I += I
        cum_U += U
        msg = 'cumulative IoU = %f' % (cum_I/cum_U)
        for n_eval_iou in range(len(eval_seg_iou_list)):
            eval_seg_iou = eval_seg_iou_list[n_eval_iou]
            seg_correct[n_eval_iou] += (I/U >= eval_seg_iou)
        if dcrf:
            I_dcrf, U_dcrf = eval_tools.compute_mask_IU(predicts_dcrf, mask)
            mean_dcrf_IoU += float(I_dcrf) / U_dcrf
            cum_I_dcrf += I_dcrf
            cum_U_dcrf += U_dcrf
            msg += '\tcumulative IoU (dcrf) = %f' % (cum_I_dcrf/cum_U_dcrf)
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct_dcrf[n_eval_iou] += (I_dcrf/U_dcrf >= eval_seg_iou)
        # print(msg)
        seg_total += 1

    # Print results
    print('Segmentation evaluation (without DenseCRF):')
    result_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        result_str += 'precision@%s = %f\n' % \
            (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou]/seg_total)
    result_str += 'overall IoU = %f; mean IoU = %f\n' % (cum_I/cum_U, mean_IoU/seg_total)
    print(result_str)
    if dcrf:
        print('Segmentation evaluation (with DenseCRF):')
        result_str = ''
        for n_eval_iou in range(len(eval_seg_iou_list)):
            result_str += 'precision@%s = %f\n' % \
                (str(eval_seg_iou_list[n_eval_iou]), seg_correct_dcrf[n_eval_iou]/seg_total)
        result_str += 'overall IoU = %f; mean IoU = %f\n' % (cum_I_dcrf/cum_U_dcrf, mean_dcrf_IoU/seg_total)
        print(result_str)
    #np.savez('IU_result_unc+.npz', IU_result)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type = str, default = '0')
    parser.add_argument('-m', type = str) # 'train' 'test'
    parser.add_argument('-n', type = str, default = 'CMSA') 
    parser.add_argument('-i', type = int, default = 800000)
    parser.add_argument('-s', type = int, default = 100000)
    parser.add_argument('-d', type = str, default = 'referit') # 'Gref' 'unc' 'unc+' 'referit'
    parser.add_argument('-c', default = False, action = 'store_true') # whether or not apply DenseCRF
    parser.add_argument('-w', type = str, default = 'deeplab') # 'resnet' 'deeplab'
    parser.add_argument('-t', type = str) # 'train' 'trainval' 'val' 'test' 'testA' 'testB'
    parser.add_argument('-lr', type = float, default = 0.00025) # start learning rate
    parser.add_argument('-bs', type = int, default = 1) # batch size
    parser.add_argument('-sfolder', type = str)
    parser.add_argument('-conv5', default = False, action = 'store_true')
    parser.add_argument('-re', type = int, default = None)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.g
    mu = np.array((104.00698793, 116.66876762, 122.67891434))

    if args.m == 'train':
        train(modelname = args.n, 
              max_iter = args.i, 
              snapshot = args.s, 
              dataset = args.d, 
              weights = args.w,
              setname = args.t,
              mu = mu,
              lr = args.lr,
              bs = args.bs,
              tfmodel_folder = args.sfolder,
              conv5 = args.conv5,
              re_iter = args.re)
    elif args.m == 'test':
        test(modelname = args.n, 
             iter = args.i, 
             dataset = args.d, 
             weights = args.w,
             setname = args.t,
             dcrf = args.c,
             mu = mu,
             tfmodel_folder = args.sfolder)
