# -*- coding:utf-8 -*-
from absl import flags
from random import shuffle, random
from gender_recog_model import *

import tensorflow as tf
import numpy as np
import os
import sys
import datetime

# Face and Gender Recognition System Based on Convolutional Neural networks, 2019
flags.DEFINE_integer("load_size", 227, "Load size")

flags.DEFINE_integer("img_size", 224, "Image size (height and width)")

flags.DEFINE_integer("img_ch", 3, "Image channels")

flags.DEFINE_integer("epochs", 200, "Total epochs")

flags.DEFINE_integer("batch_size", 32, "Batch size")

flags.DEFINE_float("lr", 0.0001, "Learngin rate")

flags.DEFINE_string("txt_path", "/yuhwan/yuhwan/Dataset/3rd_paper_dataset/[5]Gender_classification/Proposed_method/first_fold/AFAD/train.txt", "Training text path")

flags.DEFINE_string("img_path", "/yuhwan/yuhwan/Dataset/3rd_paper_dataset/[5]Gender_classification/Proposed_method/first_fold/AFAD/train/", "Training image path")

flags.DEFINE_string("test_txt_path", "/yuhwan/yuhwan/Dataset/3rd_paper_dataset/[5]Gender_classification/Proposed_method/first_fold/AFAD/test.txt", "Testing text path")

flags.DEFINE_string("test_img_path", "/yuhwan/yuhwan/Dataset/3rd_paper_dataset/[5]Gender_classification/Proposed_method/first_fold/AFAD/test/", "Testing image path")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint files path")

flags.DEFINE_string("save_checkpoint", "/yuhwan/yuhwan/checkpoint/Gender_classification/Proposed_method/[2]/first_fold/AFAD/checkpoint", "Save checkpoint files")

flags.DEFINE_string("save_graphs", "/yuhwan/yuhwan/checkpoint/Gender_classification/Proposed_method/[2]/first_fold/AFAD/graphs/", "Save training graphs")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optim = tf.keras.optimizers.Adam(FLAGS.lr)

def func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, FLAGS.img_ch)
    img = tf.image.resize(img, [FLAGS.load_size, FLAGS.load_size])
    img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch])

    if random() > 0.5:
        img = tf.image.flip_left_right(img)

    img = tf.image.per_image_standardization(img)

    lab = tf.cast(lab_list, tf.float32)

    return img, lab

def test_func(img_list, lab_list):
    
    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)

    lab = lab_list

    return img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, batch_images, batch_labels):
    with tf.GradientTape() as tape:
        logits = run_model(model, batch_images, True)

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(batch_labels, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def cal_acc(model, images, labels):
    
    logits = run_model(model, images, False)
    logits = tf.nn.sigmoid(logits)  # [batch, 1]
    # logits = tf.cast(tf.argmax(logits, 1), tf.float32)
    logits = tf.squeeze(logits, 1)

    predict = tf.cast(tf.greater(logits, 0.5), tf.float32)
    count_acc = tf.cast(tf.equal(predict, labels), tf.float32)
    count_acc = tf.reduce_sum(count_acc)

    return count_acc

def main():
    model = gender_model(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("* Restored the latest checkpoint *")

    if FLAGS.train:
        count = 0
        tr_img = np.loadtxt(FLAGS.txt_path, dtype="<U1100", skiprows=0, usecols=0)
        tr_img = [FLAGS.img_path + img for img in tr_img]
        tr_lab = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)

        test_img = np.loadtxt(FLAGS.test_txt_path, dtype="<U100", skiprows=0, usecols=0)
        test_img = [FLAGS.test_img_path + img for img in test_img]
        test_lab = np.loadtxt(FLAGS.test_txt_path, dtype=np.float32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((test_img, test_lab))
        te_gener = te_gener.map(test_func)
        te_gener = te_gener.batch(FLAGS.batch_size)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        test_idx = len(test_img) // FLAGS.batch_size
        test_iter = iter(te_gener)

        #############################
        # Define the graphs
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.save_graphs + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        val_log_dir = FLAGS.save_graphs + current_time + '/val'
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        #############################

        for epoch in range(FLAGS.epochs):

            T = list(zip(tr_img, tr_lab))
            shuffle(T)
            tr_img, tr_lab = zip(*T)
            tr_img, tr_lab = np.array(tr_img), np.array(tr_lab)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_img))
            tr_gener = tr_gener.map(func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_idx = len(tr_img) // FLAGS.batch_size
            tr_iter = iter(tr_gener)
            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)

                loss = cal_loss(model, batch_images, batch_labels)

                # save checkpoint and graphs
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=count)

                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step + 1, tr_idx, loss))

                if count % 100 == 0:
                    test_idx = len(test_img) // FLAGS.batch_size
                    test_iter = iter(te_gener)
                    acc = 0.
                    for i in range(test_idx):
                        te_img, te_label = next(test_iter)

                        acc += cal_acc(model, te_img, te_label)

                    print("====================================")
                    print("step = {}, acc = {} %".format(count, (acc / len(test_img)) * 100.))
                    print("====================================")

                    with val_summary_writer.as_default():
                        tf.summary.scalar('Acc', (acc / len(test_img)) * 100., step=count)

                if count % 1000 == 0:
                    num = int(count // 1000)
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num)
                    if not os.path.isdir(model_dir):
                        os.makedirs(model_dir)
                        print("Make {} files to save checkpoint".format(num))
                    ckpt = tf.train.Checkpoint(model=model, optim=optim)
                    ckpt_dir = model_dir + "/" + "gender_recog_{}.ckpt".format(count)

                    ckpt.save(ckpt_dir)

                count += 1

if __name__ == "__main__":
    main()