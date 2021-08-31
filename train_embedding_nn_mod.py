from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

from retrieval_model import setup_train_model
# from retrieval_model import setup_eval_model

import csv
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import random
# import pandas as pd
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
FLAGS = None



def get_batch(batch_index, batch_size, labels, f_lst):
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size
    return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])

def read_file_org(file_name):
    feat_lst = []
    label_lst = []
    with open(file_name) as fr:
        reader = csv.reader(fr, delimiter=',')
        for row in reader:
            class_label = int(float(row[-1])) - 1
            #print(class_label)
            row = row[:-1]
            s_feat = [float(i) for i in row]
            feat_lst.append(s_feat)
            label_lst.append(class_label)

    return  feat_lst, label_lst

def read_file_org_test(file_name):
    feat_lst = []
    label_lst = []
    count = 0
    with open(file_name) as fr:
        reader = csv.reader(fr, delimiter=',')
        for row in reader:
            s_feat = [float(i) for i in row]
            feat_lst.append(s_feat)
            label_lst.append(count)
    return feat_lst, label_lst

def main(_):
    
    # im_feats= tf.placeholder(tf.float32, shape=[None])
    # sent_feats = tf.placeholder(tf.float32, shape=[None])
    
    # visual = pd.read_csv('E:/UETGen/2/ganTest/featFiles/faceTest_4096.csv', header=None, sep= ',')
    # img_train = visual.loc[0].to_numpy()
    # textual = pd.read_csv('E:/UETGen/2/train_test/voiceTest.csv', header=None, sep=',')
    # voice_train = textual.loc[0].to_numpy()
    # train_label = 0
    
    im_feat_dim = 4096
    sent_feat_dim = 1024

    train_file = 'E:/saqlain/face_train.csv'
    train_file_voice = 'E:/saqlain/wav_train.csv'

    # test_file = '/home/shah/pycharm-projects/mlp/train_test/faceTest.csv'
    # test_file_voice = '/home/shah/pycharm-projects/mlp/train_test/voiceTest.csv'

    img_train, train_label = read_file_org(train_file)
    voice_train, _ = read_file_org(train_file_voice)
    # img_test, test_label = read_file_org_test(test_file)
    # img_test_voice, _ = read_file_org_test(test_file_voice)

    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    train_label = le.transform(train_label)


    print("Train file length", len(img_train))
    #print("Test file length", len(img_test))
    print("Train label length", len(train_label))
    #print("Test label length", len(test_label))

    # mean_data_img_train = np.mean(img_train, axis=0)
    # mean_data_voice_train = np.mean(voice_train, axis=0)

    combined = list(zip(img_train, voice_train, train_label))
    random.shuffle(combined)
    img_train[:], voice_train, train_label[:] = zip(*combined)



    # Load data.
    steps_per_epoch = len(voice_train) // FLAGS.batch_size
    # num_steps = steps_per_epoch * FLAGS.max_num_epoch

    # im_feat_plh = tf.placeholder(tf.float32, shape=[None])
    # sent_feat_plh = tf.placeholder(tf.float32, shape=[None])
    # Setup placeholders for input variables.
    im_feat_plh = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, im_feat_dim])
    sent_feat_plh = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, sent_feat_dim])
    label_plh = tf.placeholder(tf.int64, shape=(None), name='labels')
    train_phase_plh = tf.placeholder(tf.bool)

    #exit()

    # Setup training operation.
    softmax_loss, central_loss, loss = setup_train_model(im_feat_plh, sent_feat_plh, train_phase_plh, label_plh, FLAGS)
    print('')
    # Setup optimizer.
    global_step = tf.Variable(0, trainable=False)
    init_learning_rate = 0.0001
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                               steps_per_epoch, 0.769, staircase=True)
    optim = tf.train.AdamOptimizer(learning_rate)
    
    # gradients, variables = zip(*optim.compute_gradients(loss))
    # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    # optim = optim.apply_gradients(zip(gradients, variables))
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optim.minimize(loss, global_step=global_step)

    # Setup model saver.
    saver = tf.train.Saver(save_relative_paths=False)

    num_train_samples = len(img_train)
    num_of_batches = (num_train_samples // FLAGS.batch_size)

    # ax = plt.subplot()    
    totl_loss = []
    soft_loss = []
    cent_loss = []
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5):
            for idx in range(num_of_batches):
                im_feats, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, img_train)
                sent_feats, _ = get_batch(idx, FLAGS.batch_size, train_label, voice_train)

                feed_dict = {
                        im_feat_plh : im_feats ,
                        sent_feat_plh : sent_feats ,
                        label_plh : batch_labels,
                        train_phase_plh : True,
                }
                
                [_, soft, cent, loss_val] = sess.run([train_step, softmax_loss, central_loss, loss], feed_dict = feed_dict)
                
                
                soft_loss.append(soft)
                cent_loss.append(cent)
                totl_loss.append(loss_val)
                
                print('Epoch: %d Step: %d Loss: %f' % (i , idx, loss_val))
            print('Saving checkpoint at step %d' % i)
            saver.save(sess, FLAGS.save_dir, global_step = global_step)
    plt.figure(0)
    plt.title('Total Loss')
    plt.plot(totl_loss)
    print("Minimum total loss {}".format(min(totl_loss)))
    plt.figure(1)
    plt.title('Softmax Loss')
    plt.plot(soft_loss)
    print("Minimum soft loss {}".format(min(soft_loss)))
    plt.figure(2)
    plt.title('Central Loss')
    plt.plot(cent_loss)
    print("Minimum central loss {}".format(min(cent_loss)))


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser()
    # Dataset and checkpoints.
    parser.add_argument('--image_feat_path', type=str, help='Path to the image feature mat file.')
    parser.add_argument('--sent_feat_path', type=str, help='Path to the sentence feature mat file.')
    parser.add_argument('--save_dir', type=str, default='./orgplot_3/', help='Directory for saving checkpoints.')
    parser.add_argument('--restore_path', type=str, help='Path to the restoring checkpoint MetaGraph file.')
    # Training parameters.
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--sample_size', type=int, default=2, help='Number of positive pair to sample.')
    parser.add_argument('--max_num_epoch', type=int, default=20, help='Max number of epochs to train.')
    parser.add_argument('--num_neg_sample', type=int, default=10, help='Number of negative example to sample.')
    parser.add_argument('--margin', type=float, default=0.5, help='Margin.')
    parser.add_argument('--im_loss_factor', type=float, default=1.5,
                        help='Factor multiplied with image loss. Set to 0 for single direction.')
    parser.add_argument('--sent_only_loss_factor', type=float, default=0.05,
                        help='Factor multiplied with sent only loss. Set to 0 for no neighbor constraint.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
