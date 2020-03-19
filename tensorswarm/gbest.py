import tensorflow as tf
from utils import *

@tf.function
def gbest(f, bounds, particles=1, dim=3,
            iters=1000,
            w=0.9, c1=0.7, c2=0.7):
    # Init
    x, v = init_particles(particles, dim, bounds)
    pb = f(x)
    pb_x = tf.identity(x)
    gb = tf.constant(float('Inf'), dtype=tf.float16)
    gb_x = tf.zeros([dim], dtype=tf.float16)

    # Constants

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
        with tf.name_scope("globalbest"):
            gb_i = tf.argmin(pb)
            gb = pb[gb_i]
            gb_x = pb_x[gb_i]

        # Movement
        with tf.name_scope("movement"):
            r1 = tf.random.uniform([particles, dim], dtype=tf.float16)
            r2 = tf.random.uniform([particles, dim], dtype=tf.float16)

            cog = r1*(pb_x-x)
            soc = r2*(gb_x-x)

            v = w*v + c1*cog + c2*soc

            x += v

        # Summary
        tf.summary.scalar("global_best", gb, step=i)
        log_diversity(x, i)

        # Stopping condition
        with tf.name_scope("stopping_cond"):
            i += 1

    return gb, gb_x, i
if __name__ == '__main__':
    from functions import *
    import time

    particles = 3
    dim = 3
    iters = 1000

    writer = tf.summary.create_file_writer("../logs/gbest_{}".format(time.strftime("%Y-%m-%d_%H:%M:%S")))
    with writer.as_default():
        val, x, i  = gbest(f_abs, (-100,100), particles=particles, dim=dim, iters=iters)
        print(f"{val} @ {x} - step: {i}")
    