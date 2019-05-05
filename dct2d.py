# @Dutmedia Lab, Heyi 2019/5/5

import tensorflow as tf
import numpy as np
import math

# This is matlab version fft operation
def matlab_fft(y):
    y_t = tf.transpose(y, [0,2,1])
    f_y = tf.fft(tf.cast(y_t, tf.complex64))
    f_y = tf.transpose(f_y, [0,2,1])
    return f_y

def matlab_ifft(y):
    y_t = tf.transpose(y, [0,2,1])
    f_y = tf.ifft(tf.cast(y_t, tf.complex64))
    f_y = tf.transpose(f_y, [0,2,1])
    return f_y   

def dct2d_core(x):

    N = tf.shape(x)[0]
    n = tf.shape(x)[1]
    m = tf.shape(x)[2]

    y = tf.reverse(x, axis = [1])
    y = tf.concat([x, y], axis = 1)
    f_y = matlab_fft(y)
    f_y = f_y[:, 0:n, :]

    t = tf.complex(tf.constant([0.0]), tf.constant([-1.0])) * tf.cast(tf.linspace(0.0, tf.cast(n-1, tf.float32), n), tf.complex64)
    t = t * tf.cast(math.pi / (2.0 * tf.cast(n, tf.float64)), tf.complex64)
    t = tf.exp(t) / tf.cast(tf.sqrt(2.0 * tf.cast(n, tf.float64)), tf.complex64)

    # since tensor obejct does not support item assignment, we have to concat a new tensor
    t0 = t[0] / tf.cast(tf.sqrt(2.0), tf.complex64)
    t0 = tf.expand_dims(t0, 0)
    t = tf.concat([t0, t[1:]], axis = 0)
    t = tf.expand_dims(t, -1)
    t = tf.expand_dims(t, 0)
    W = tf.tile(t, [N,1,m])

    dct_x = W * f_y
    dct_x = tf.cast(dct_x, tf.complex64)
    dct_x = tf.real(dct_x)

    return dct_x


def idct2d_core(x):

    N = tf.shape(x)[0]
    n = tf.shape(x)[1]
    m = tf.shape(x)[2]

    temp_complex = tf.complex(tf.constant([0.0]), tf.constant([1.0]))
    t = temp_complex * tf.cast(tf.linspace(0.0, tf.cast(n-1, tf.float32), n), tf.complex64)
    t = tf.cast(tf.sqrt(2.0 * tf.cast(n, tf.float64)), tf.complex64) * tf.exp(t * tf.cast(math.pi / (2.0 * tf.cast(n, tf.float64)), tf.complex64))

    t0 = t[0] * tf.cast(tf.sqrt(2.0), tf.complex64)
    t0 = tf.expand_dims(t0, 0)
    t = tf.concat([t0, t[1:]], axis = 0)
    t = tf.expand_dims(t, -1)
    t = tf.expand_dims(t, 0)
    W = tf.tile(t, [N,1,m])

    x = tf.cast(x, tf.complex64)
    yy_up = W * x
    temp_complex = tf.complex(tf.constant([0.0]), tf.constant([-1.0]))
    yy_down = temp_complex * W[:, 1:n, :] * tf.reverse(x[:,1:n, :], axis = [1])
    yy_mid = tf.cast(tf.zeros([N, 1, m]), tf.complex64)
    yy = tf.concat([yy_up, yy_mid, yy_down], axis = 1)
    y = matlab_ifft(yy)
    y = y[:, 0:n, :]
    y = tf.real(y)

    return y


def dct2d(x):
    x = dct2d_core(x)
    x = tf.transpose(x, [0,2,1])
    x = dct2d_core(x)
    x = tf.transpose(x, [0,2,1])
    return x

def idct2d(x):
    x = idct2d_core(x)
    x = tf.transpose(x, [0,2,1])
    x = idct2d_core(x)
    x = tf.transpose(x, [0,2,1])
    return x


if __name__ == "__main__":

    import cv2

    x1 = cv2.imread('test001.png', 0)
    x1 = x1 / 255
    x1 = x1[np.newaxis, ...]
    x2 = cv2.imread('test002.png', 0)
    x2 = x2 / 255
    x2 = x2[np.newaxis, ...]
    x3 = cv2.imread('test003.png', 0)
    x3 = x3 / 255
    x3 = x3[np.newaxis, ...]
    x = np.concatenate((x1,x2,x3), axis = 0)


    x_in = tf.placeholder(tf.float64, shape=(None, None, None), name = 'input')
    dct2d_x = dct2d(x_in)
    idct2d_x = idct2d(dct2d_x)

    with tf.Session() as sess:
        with tf.device('CPU:0'):
            dct2d_x_out, idct2d_x_out = sess.run([dct2d_x, idct2d_x], feed_dict = {x_in: x})
            

    error = x - idct2d_x_out
    error = np.mean(error)
    print(error)

    idct2d_x_out = np.squeeze(idct2d_x_out)
    x1 = idct2d_x_out[0] * 255
    x2 = idct2d_x_out[1] * 255
    x3 = idct2d_x_out[2] * 255

    cv2.imwrite('1.png', x1.astype('uint8'))
    cv2.imwrite('2.png', x2.astype('uint8'))
    cv2.imwrite('3.png', x3.astype('uint8'))

    print('done')



