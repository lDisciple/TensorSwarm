import tensorflow as tf
from ..utils import *

@tf.function
def lpredator_prob(f, bounds, particles=6, predators=1, dim=3,
            iters=1000, dtype=tf.float64,
            fear_factor=0.8, prey_vmax=0.04, fear_dist=0.1, pred_vmax=0.02,
            w=0.9, c1=0.7, c2=0.7, c3=0.7):
    # Init
    x, v = init_particles(particles, dim, bounds, dtype=dtype)
    pb = f(x)
    pb_x = tf.identity(x)

    x_pred = init_predators(predators, dim, bounds, dtype=dtype)

    # Constants
    bound_dist = bounds[1] - bounds[0]
    alpha = prey_vmax*bound_dist # Velocity at distance 0 for the predator
    """
    4/b is the distance where the prey starts running with a reasonable magnitude.
    We scale it based on dimensions as more dimensions means bigger distances.
    """
    beta = (4/(fear_dist*bound_dist)) / tf.math.sqrt(tf.cast(dim, dtype))


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

        # Neighbourhood best
        with tf.name_scope("neighbourhoodbest"):
            choices = tf.gather_nd(pb, nb_mask)
            choices = tf.argmin(choices, axis=1, output_type=tf.int32)
            indices = tf.math.floormod(prange + choices - 1, particles)
            nb_x = tf.gather(pb_x, indices)

        # Movement
        with tf.name_scope("movement"):
            r1 = tf.random.uniform([particles, dim], dtype=dtype)
            r2 = tf.random.uniform([particles, dim], dtype=dtype)

            fear = fear_component(x, x_pred, alpha, beta, fear_factor)
            cog  = r1*(pb_x-x)
            soc  = r2*(nb_x-x)

            v = w*v + c1*cog + c2*soc + c3*fear

            x += v

            # preb_nb_x = nearest_particles(x_pred, nb_x)
            selected = distance_tournament_select(x_pred, nb_x)
            x_pred += predator_velocity(x_pred, selected, pred_vmax)

        # Summary
        log_boundary_violations(x, bounds, i)
        log_personalbest(pb, i)
        log_diversity(x, i)

        # Stopping condition
        with tf.name_scope("stopping_cond"):
            i += 1

    best_i = tf.argmin(pb)
    return pb[best_i], pb_x[best_i], i

if __name__ == '__main__':
    from ..functions import *
    import time

    particles = 30
    predators = 2
    dim = 30
    iters = 5000
    dtype=tf.float64

    writer = tf.summary.create_file_writer("../logs/lpredator_prob_{}".format(time.strftime("%Y-%m-%d_%H:%M:%S")))
    with writer.as_default():
        val, x, i  = lpredator_prob(f_abs, (-100,100),
            particles=particles, dim=dim,
            predators=predators,
            iters=iters, dtype=dtype)
        print(f"{val} @ {x} - step: {i}")
