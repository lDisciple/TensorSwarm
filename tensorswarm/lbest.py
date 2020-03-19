import tensorflow as tf
from utils import *

@tf.function
def lbest(f, bounds, particles=1, dim=3,
            iters=1000,
            w=0.9, c1=0.7, c2=0.7):
    # Init
    x, v = init_particles(particles, dim, bounds)
    pb = f(x)
    pb_x = tf.identity(x)
    nb_x = tf.zeros([dim], dtype=tf.float16)

    # Constants
    prange = tf.range(particles)
    
    nb_mask = tf.stack((
        tf.math.floormod(prange+particles-1, particles),
        prange,
        tf.math.floormod(prange+1, particles)
    ), 1)
    nb_mask = tf.reshape(nb_mask, (particles, 3, 1))

    # Main loop
    i = tf.constant(0, tf.int64) 
    while iters > i:
        # Fitness
        with tf.name_scope("fitness"):
            fit = bounded_fitness(f, x, bounds)

        # Update personal best
        with tf.name_scope("personalbest"):
            pb, pb_x = eval_personalbest(fit, x, pb, pb_x)

        # Global best
        with tf.name_scope("neighbourhoodbest"):
            choices = tf.gather_nd(pb, nb_mask)
            choices = tf.argmin(choices, axis=1, output_type=tf.int32)
            indices = tf.math.floormod(prange + choices - 1, particles)
            nb_x = tf.gather(x, indices)

        # Movement
        with tf.name_scope("movement"):
            r1 = tf.random.uniform([particles, dim], dtype=tf.float16)
            r2 = tf.random.uniform([particles, dim], dtype=tf.float16)

            cog = r1*(pb_x-x)
            soc = r2*(nb_x-x)

            v = w*v + c1*cog + c2*soc

            x += v

        # Summary
        log_personalbest(pb, i)
        log_diversity(x, i)

        # Stopping condition
        with tf.name_scope("stopping_cond"):
            i += 1

    best_i = tf.argmin(pb)
    return pb[best_i], pb_x[best_i], i
if __name__ == '__main__':
    from functions import *
    import time

    particles = 3
    dim = 3
    iters = 1000

    writer = tf.summary.create_file_writer("../logs/lbest_{}".format(time.strftime("%Y-%m-%d_%H:%M:%S")))
    with writer.as_default():
        val, x, i  = lbest(f_abs, (-100,100), particles=particles, dim=dim, iters=iters)
        print(f"{val} @ {x} - step: {i}")
    