import tensorflow as tf
from gen_img import get_sample


def main():
    tfrecord_root='re'
    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_root + '/chamo_paint.tfrecord')
    for i in range(100000):
        try:
            img, cur_id = get_sample(0)
            size = img.size
            image_data = tf.gfile.FastGFile('./re/chamo.jpg', 'rb').read()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[cur_id])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
                }))
            tfrecord_writer.write(example.SerializeToString())
        except:
            print("file errir")
    tfrecord_writer.close()

main()
