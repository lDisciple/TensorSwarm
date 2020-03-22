import tensorflow as tf

@tf.function
def init_particles(particles, dim, bounds, dtype=tf.float16):
    x = tf.random.uniform([particles, dim], dtype=dtype,
        minval=bounds[0], maxval=bounds[1])
    v = tf.zeros([particles, dim], dtype=dtype)

    return x, v

def init_predators(predators, dim, bounds, dtype=tf.float16):
    return tf.random.uniform([predators, dim], dtype=dtype,
                minval=bounds[0], maxval=bounds[1])

@tf.function
def boundary_violations(x, bounds):
    lower_violations = tf.reduce_any(x < bounds[0], axis=1)
    upper_violations = tf.reduce_any(x > bounds[1], axis=1)
    return tf.logical_or(lower_violations, upper_violations)

@tf.function
def bounded_fitness(f, x, bounds):
    fit = f(x)
    bvs = boundary_violations(x, bounds)
    fit = tf.where(bvs, x=tf.constant(float('Inf'), fit.dtype), y=fit) # Set out of bounds to inf

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
def log_boundary_violations(x, bounds, step):
    with tf.name_scope("boundary_violations"):
        bvs = boundary_violations(x, bounds)
        out = tf.reduce_sum(tf.cast(bvs, tf.float16))
        tf.summary.scalar("particles", out, step=step)
        tf.summary.scalar("percentage", out/bvs.shape[-1], step=step)

@tf.function
def log_personalbest(pb, step):
    with tf.name_scope("personal_best"):
        tf.summary.scalar("min", tf.math.reduce_min(pb), step=step)
        tf.summary.scalar("max", tf.math.reduce_max(pb), step=step)
        tf.summary.scalar("mean", tf.math.reduce_mean(pb), step=step)

@tf.function
def nearest_particle(x, swarm):
    # tf.print("X:", x)
    # tf.print("Found:", swarm[tf.argmin(tf.norm(swarm-x, axis=1))])
    return swarm[tf.argmin(tf.norm(swarm-x, axis=1))]

@tf.function
def nearest_particles(x, swarm):
    # tf.print("Swarm:", swarm)
    return tf.map_fn(lambda x: nearest_particle(x, swarm), x)

@tf.function
def fear_component(x, x_pred, alpha, beta, fear_factor):
    r = tf.random.uniform(x.shape, dtype=x.dtype)
    target = x - nearest_particles(x, x_pred) # Run from nearest predators
    d = tf.norm(target, axis=1)
    d = tf.reshape(d, (-1,1))
    D = alpha*tf.exp(-beta*d)
    comp = r*D*target/d
    with tf.name_scope("bravery"):
        mood = tf.random.uniform(x.shape[:1], dtype=x.dtype)
        cond = tf.reshape(mood < fear_factor, (-1, 1)) # Runners = True
        comp = tf.where(cond, x=comp, y=0)

    return comp

@tf.function
def distance_tournament_select(x, swarm, k=tf.constant(2)):
    indices = tf.random.uniform([x.shape[0], k],
                                maxval=swarm.shape[0],
                                dtype=tf.int32)
    selected = tf.gather(swarm, indices)
    x = tf.reshape(x, (-1, 1, x.shape[1]))
    d = tf.norm(selected-x, axis=2)
    left=d[:, 0]
    right=d[:, 1]
    b = tf.reshape(left < right, (-1,1))
    return tf.where(b, x=selected[:,0], y=selected[:,1])

@tf.function
def predator_velocity(x, best, vmax):
    r = tf.random.uniform(x.shape, maxval=vmax, dtype=x.dtype)
    return r*(best-x)
