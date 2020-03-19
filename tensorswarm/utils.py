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