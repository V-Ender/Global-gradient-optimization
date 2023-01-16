import numpy as np
import numdifftools as nd


def minimum(f, lb, rb, num_x=10000):
    step = (rb - lb) / num_x
    x_arr = np.arange(lb, rb, step)
    min_x = lb
    for x in x_arr:
        if f(x) < f(min_x):
            min_x = x
    return min_x


def t_bounds(p, d, a):
    dim = len(p)
    pos_t, neg_t = np.inf, -np.inf
    for i in range(dim):
        t_tmp = (a - p[i]) / d[i]
        #print(p[i])
        if (t_tmp >= 0) and (t_tmp < pos_t):
            pos_t = t_tmp
        if (t_tmp < 0) and (t_tmp > neg_t):
            neg_t = t_tmp

        t_tmp = (-a - p[i]) / d[i]
        if (t_tmp >= 0) and (t_tmp < pos_t):
            pos_t = t_tmp
        if (t_tmp < 0) and (t_tmp > neg_t):
            neg_t = t_tmp

    return (neg_t, pos_t)


def global_gradient_descent_2d(f, num_iters, size=1, steps=10000, tolerance=None, bound=32, verbose=False):
    x_tmp = np.random.uniform(-bound, bound, size)
    print(x_tmp)
    f_tmp = f(x_tmp)
    fs = []
    fs.append(f_tmp)

    if verbose:
        print("f is: " + str(f_tmp))

    for i in range(num_iters):
        grad = nd.Gradient(f)(x_tmp)

        p = x_tmp
        d = grad
        t_min, t_max = t_bounds(p, d, bound)

        phi = lambda t: p + t * d
        f_tmp = lambda t: f(phi(t))
        min = minimum(f_tmp, float(t_min), float(t_max), steps)
        x_tmp = p + min * d

        if tolerance is not None:
            if abs(f(x_tmp) - fs[-1]) < tolerance:
                break

        if verbose:
            print(x_tmp, f(x_tmp))

        fs.append(f(x_tmp))

    return x_tmp, f(x_tmp)

def vanilla_gradient_descent(f, num_iters, learning_rate, bound=32, verbose=False):
    x_tmp = np.random.uniform(-bound, bound)
    y_tmp = np.random.uniform(-bound, bound)
    f_tmp = f([x_tmp, y_tmp])
    fs=[]
    fs.append(f_tmp)

    if(verbose):
        print("The cost is: "+str(f_tmp))

    for i in range(num_iters):
        grad = nd.Gradient(f)([x_tmp, y_tmp])
        x_tmp = x_tmp - learning_rate * grad[0]
        y_tmp = y_tmp - learning_rate * grad[1]
        f_tmp = f([x_tmp, y_tmp])
        if(f_tmp > 10000):
            print("Cost explosion, trying with lower learning rate")
            learning_rate=learning_rate/10
            print("Learning rate changed to "+str(round(learning_rate,2)))
        if(verbose):
            if(i%10==0):
                print("The cost is: "+str(f_tmp))
        fs.append(f_tmp)
        print(x_tmp, y_tmp, f([x_tmp, y_tmp]))

    x_opt = x_tmp
    y_opt = y_tmp
    return x_opt, y_opt, fs


def ackley_f(x):
    a, b, c = 20, 0.2, 2 * np.pi
    return -a * np.exp(-b * np.sqrt((x**2).mean())) - np.exp(np.cos(c * x).mean()) + a + np.exp(1)


#print(ackley_f(np.array([-9.40800000e-01, -5.81627807e-04,  9.63522812e-04])))
x, ans = global_gradient_descent_2d(ackley_f, 20, 3, 100000, verbose=True)
print(ans)
#print('Simple')
#x, y, arr = vanilla_gradient_descent(ackley_f, 1000, 0.001)

