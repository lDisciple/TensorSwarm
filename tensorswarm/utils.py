import tensorflow as tf

@tf.function
def init_particles(particles, dim, bounds):
    x = tf.random.uniform([particles, dim], dtype=tf.float16,
        minval=bounds[0], maxval=bounds[1])
    v = tf.zeros([particles, dim], dtype=tf.float16)

    return x, v

def init_predators(predators, dim, bounds):
    return tf.random.uniform([predators, dim], dtype=tf.float16,
                minval=bounds[0], maxval=bounds[1])

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

@tf.function
def nearest_particle(x, swarm):
    return swarm[tf.argmin(tf.norm(swarm-x, axis=1))]

@tf.function
def nearest_particles(x, swarm):
    return tf.map_fn(lambda x: nearest_particle(x, swarm), x)

@tf.function
def fear_component(x, x_pred, alpha, beta, fear_factor):
    r = tf.random.uniform(x.shape, dtype=tf.float16)
    target = x - nearest_particles(x, x_pred) # Run from nearest predators
    d = tf.norm(target, axis=1)
    d = tf.reshape(d, (-1,1))
    D = alpha*tf.exp(-beta*d)
    comp = r*D*target/d
    with tf.name_scope("bravery"):
        mood = tf.random.uniform(x.shape[:1], dtype=tf.float16)
        cond = tf.reshape(mood < fear_factor, (-1, 1)) # Runners = True
        comp = tf.where(cond, x=comp, y=0)

    return comp

@tf.function
def predator_velocity(x, best, vmax):
    r = tf.random.uniform(x.shape, maxval=vmax, dtype=tf.float16)
    return r*(best-x)
