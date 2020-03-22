import tensorflow as tf
from utils import *

@tf.function
def gpredator(f, bounds, particles=6, predators=1, dim=3,
            iters=1000,
            fear_factor=0.8, prey_vmax=0.04, fear_dist=0.1, pred_vmax=0.008,
            w=0.9, c1=0.7, c2=0.7, c3=0.7):
    # Init
    x, v = init_particles(particles, dim, bounds)
    pb = f(x)
    pb_x = tf.identity(x)
    gb = tf.constant(float('Inf'), dtype=tf.float16)
    gb_x = tf.zeros([dim], dtype=tf.float16)

    x_pred = init_predators(predators, dim, bounds)

    # Constants
    bound_dist = bounds[1] - bounds[0]
    alpha = prey_vmax*bound_dist # Velocity at distance 0 for the predator
    """
    4/b is the distance where the prey starts running with a reasonable magnitude.
    We scale it based on dimensions as more dimensions means bigger distances.
    """
    beta = (4/(fear_dist*bound_dist)) / tf.math.sqrt(tf.cast(dim, tf.float16))

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

            fear = fear_component(x, x_pred, alpha, beta, fear_factor)
            cog  = r1*(pb_x-x)
            soc  = r2*(gb_x-x)

            v = w*v + c1*cog + c2*soc + fear

            x += v
            x_pred += predator_velocity(x_pred, gb, pred_vmax)

        # Summary
        tf.summary.scalar("global_best", gb, step=i)
        log_personalbest(pb, i)
        log_diversity(x, i)

        # Stopping condition
        with tf.name_scope("stopping_cond"):
            i += 1

    return gb, gb_x, i
if __name__ == '__main__':
    from functions import *
    import time

    particles = 30
    predators = 2
    dim = 30
    iters = 5000

    writer = tf.summary.create_file_writer("../logs/gpredator_{}".format(time.strftime("%Y-%m-%d_%H:%M:%S")))
    with writer.as_default():
        val, x, i  = gpredator(f_abs, (-100,100),
            particles=particles, dim=dim,
            predators=predators,
            iters=iters)
        print(f"{val} @ {x} - step: {i}")
