import tensorflow as tf

@tf.function
def init_particles(particles, dim, bounds):
    x = tf.random.uniform([particles, dim], dtype=tf.float16,
        minval=bounds[0], maxval=bounds[1])
    v = tf.zeros([particles, dim], dtype=tf.float16)
    
    return x, v

@tf.function
def bounded_fitness(f, x, bounds):
    fit = f(x)

    lower_fails = tf.reduce_any(x < bounds[0], axis=1)
    upper_fails = tf.reduce_any(x > bounds[1], axis=1)
    boundary_fails = tf.logical_or(lower_fails, upper_fails)
    fit = tf.where(boundary_fails, x=tf.constant(float('Inf'), tf.float16), y=fit) # Set out of bounds to inf

    return fit

@tf.function
def eval_personalbest(fit, x, pb, pb_x):
    cond = pb > fit
    pb = tf.where(pb > fit, x=fit, y=pb)
    cond = tf.reshape(cond, (-1, 1))
    pb_x = tf.where(cond, x=x, y=pb_x)

    return pb, pb_x

@tf.function
def diversity(x):
    a = tf.math.reduce_mean(x, axis=0)
    return tf.norm(x-a, axis=1)

@tf.function
def log_diversity(x, step):
    with tf.name_scope("diversity"):
        div = diversity(x)
        tf.summary.scalar("min", tf.math.reduce_min(div), step=step)
        tf.summary.scalar("max", tf.math.reduce_max(div), step=step)
        tf.summary.scalar("mean", tf.math.reduce_mean(div), step=step)

@tf.function
def log_personalbest(pb, step):
    with tf.name_scope("personal_best"):
        tf.summary.scalar("min", tf.math.reduce_min(pb), step=step)
        tf.summary.scalar("max", tf.math.reduce_max(pb), step=step)
        tf.summary.scalar("mean", tf.math.reduce_mean(pb), step=step)