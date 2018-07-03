import tensorflow as tf
import os
import mobilenet.mobilenet_v2 as mnv2
import time
import matplotlib.pyplot as plt
slim = tf.contrib.slim

img_mean = [125, 125, 125]
img_size=224

def _mean_image_subtraction(image):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(img_mean) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= img_mean[i]
    return tf.concat(axis=2, values=channels)

def check_imgs(images, labels):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        image_batch_v, label_batch_v = sess.run([images, labels])
        for k in range(len(image_batch_v)):
            processed_img = image_batch_v[k]
            print(label_batch_v[k])
            show_img = processed_img + img_mean
            show_img = abs(show_img) / 256.0
            plt.imshow(show_img)
            plt.show()
        coord.request_stop()
        coord.join(threads)

def get_raw_img(tfrecord_addr):
    file_list = os.listdir(tfrecord_addr)
    tfrecord_list=[]
    for file_name in file_list:
        if file_name.find('.tfrecord'):
            tfrecord_list.append(tfrecord_addr+'/'+file_name)
    filename_queue = tf.train.string_input_producer(tfrecord_list)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'img_width': tf.FixedLenFeature([], tf.int64),
                                           'img_height': tf.FixedLenFeature([], tf.int64),
                                       })
    height = tf.cast(features['img_height'], tf.int32)
    width = tf.cast(features['img_width'], tf.int32)
    img_data_jpg = tf.image.decode_jpeg(features['img_raw'])
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)
    image = img_data_jpg
    channel = 3
    image = tf.reshape(image, [height, width, channel])
    id = tf.cast(features['label'], tf.int32)
    depth = 4
    label=tf.one_hot(id, depth)
    #label=tf.Print(label, [label], "chamo:")
    return image, label

image, label=get_raw_img('./re')
image.set_shape([img_size, img_size, 3])
image = tf.to_float(image)
image=_mean_image_subtraction(image)
batchsize=32
images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batchsize,
            num_threads=1,
            capacity=5 * batchsize,
            min_after_dequeue=3 * batchsize
        )
#check_imgs(images, labels)
with tf.contrib.slim.arg_scope(mnv2.training_scope(is_training=True)):
    logits, endpoint = mnv2.mobilenet(images, num_classes=4, reuse=tf.AUTO_REUSE)
#labels=tf.Print(labels, [labels], "labels:",summarize=32)
#labels=tf.Print(labels, [tf.sigmoid(logits)], "logits:",summarize=32)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels =labels, logits=logits))
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = slim.learning.create_train_op(loss,optimizer)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    #saver.restore(sess, './output/chamo_4000.000000_0.000018/chamo.ckpt')
    writer = tf.summary.FileWriter("logs/", sess.graph)
    i =-1
    while True:
        before_time = time.perf_counter()
        i=i+1
        re=sess.run([train_step,loss])
        after_time = time.perf_counter()
        step_time = after_time - before_time
        if i % 200 == 0:
            print("[step: %f][train loss: %f][time: %f]" % (i, re[1], step_time))
            #print(re[3])    
            #print(re[2])
            if i % (200) == 0:
                output_name='output/chamo_%f_%f' % (i, re[1])
                os.system('mkdir '+output_name)
                saver.save(sess, output_name+'/chamo.ckpt')
    coord.request_stop()
    coord.join(threads)
