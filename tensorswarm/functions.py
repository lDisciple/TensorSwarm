import tensorflow as tf
import math

@tf.function
def f_abs(x):
    return tf.reduce_sum(tf.abs(x), axis=1)
