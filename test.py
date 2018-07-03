import tensorflow as tf
import os
import mobilenet.mobilenet_v2 as mnv2
from gen_img import get_sample
import matplotlib.pyplot as plt

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

image_raw_data = tf.placeholder(tf.string, None)
img_data = tf.image.decode_jpeg(image_raw_data)
image = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)
img_show=image
image.set_shape([img_size, img_size, 3])
image = tf.to_float(image)
image=_mean_image_subtraction(image)
image=tf.expand_dims(image,[0])

with tf.contrib.slim.arg_scope(mnv2.training_scope(is_training=False)):
    logits, endpoint = mnv2.mobilenet(image, num_classes=4)
#logits=tf.Print(logits,[logits],'logits: ',summarize=32)
#logits=tf.sigmoid(logits)
#logits=tf.nn.softmax(logits)
#logits = tf.Print(logits, [logits], 'softmax: ', summarize=32)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, './output/chamo_3000.000000_0.002084/chamo.ckpt')
    while True:
        img, cur_id = get_sample(0)
        image_raw_data_jpg = tf.gfile.FastGFile('./re/chamo.jpg', 'rb').read()
        re = sess.run([logits, img_show], feed_dict={image_raw_data: image_raw_data_jpg})
        print(re[0][0])
        show_img = re[1]
        plt.imshow(show_img)
        plt.show()
