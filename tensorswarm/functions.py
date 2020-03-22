import tensorflow as tf
import math

pi = tf.constant(math.pi, tf.float16)
e = tf.constant(math.e, tf.float16)

@tf.function
def f_abs(x):
    return tf.reduce_sum(tf.abs(x), axis=1)

@tf.function
def f_ackley(x):
    dimi = tf.constant(1, tf.float16)/x.shape[-1]
    t1 = tf.reduce_sum(tf.math.square(x), axis=1)
    t1 = -20*tf.math.exp(-0.2*tf.math.sqrt(dimi*t1))

    t2 = tf.math.cos(2*pi*x)
    t2 = tf.math.exp(dimi*tf.reduce_sum(t2, axis=1))

    return t1 - t2 + 20 + e

@tf.function
def f_quadric(x):
    return tf.reduce_sum(tf.math.square(tf.math.cumsum(x, axis=1)), axis=1)

@tf.function
def f_step(x):
    return tf.reduce_sum(tf.math.square(tf.math.floor(x+0.5)), axis=1)

def gen_weierstrass(dim):
    a = tf.constant(0.5, tf.float32)
    b = tf.constant(3, tf.float32)
    i = tf.range(1, 21, dtype=tf.float32)
    ai = tf.math.pow(a, i)
    bi = tf.math.pow(b, i)
    abi = tf.stack((ai, bi), axis=1)

    pi = tf.constant(math.pi, tf.float32)
    half = tf.constant(0.5, tf.float32)
    two = tf.constant(2, tf.float32)

    t2 = tf.math.cos(pi*bi)
    t2 = ai*t2
    t2 = dim*tf.math.reduce_sum(t2)

    @tf.function
    def f_weierstrass(x):
        x = tf.cast(x, tf.float32)

        f = lambda abi: abi[0]*tf.math.cos(two*pi*abi[1]*(x+half))
        t1 = tf.map_fn(f, abi)
        t1 = tf.reduce_sum(t1, axis=(0,2))
        return t1 - t2
    return f_weierstrass

def get_function_list(dim):
    return [f_abs, f_step, f_ackley, f_quadric, gen_weierstrass(dim)]

def get_bounds_list(dim):
    bounds = [(-100,100), (-100,100), (-32.768,32.768), (-100,100), (-0.5,0.5)]
    return tf.convert_to_tensor(bounds, dtype=tf.float16)

if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    funcs = get_function_list(2)
    bounds = get_bounds_list(2)

    test_pts = tf.convert_to_tensor([(0,0), (1/3, 1/3)], dtype=tf.float16)


    #
    # for f in funcs:
    #     tf.print(f.__name__, "\n----------------------------------------")
    for f, b in zip(funcs, bounds):
        tf.print(f.__name__, "\n----------------------------------------")


        draw_pts = 200
        l = tf.linspace(b[0], b[1], draw_pts)
        x,y = tf.meshgrid(l, l)
        xf = tf.reshape(x, [-1])
        yf = tf.reshape(y, [-1])
        pts = tf.stack((xf,yf), axis=1)
        z = f(pts)

        mini = tf.argmin(z)
        tf.print(pts[mini], "->", z[mini])

        z = tf.reshape(z, (draw_pts, draw_pts)).numpy()


        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_title(f.__name__)
        # Plot the surface.
        surf = ax.plot_surface(x.numpy(), y.numpy(), z, cmap='coolwarm',
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(b[0], b[1])

        # # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        results = f(test_pts)
        for x, r in zip(test_pts, results):
            tf.print(x, "->", r)
        tf.print("----------------------------------------")

        plt.show()
