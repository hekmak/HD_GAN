import os
import matplotlib.pyplot as plt
import tensorflow as tf

def save_images_from_event(fn, tag, output_dir='./event_preds'):
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    plt.imsave(arr= im, fname= output_fn )
                    count += 1  



if __name__ == '__main__':
    save_images_from_event('./logs/2019_9_15_15_40/events.out.tfevents.1568562013.09607ce57272',
    'GPU0/Prediction/fake_image/image/0', output_dir='./event_preds_2')