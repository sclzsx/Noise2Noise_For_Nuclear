def mytest_tf_gpu():
    import tensorflow as tf

    with tf.device('/cpu:0'):
        a = tf.constant([1.0,2.0,3.0],shape=[3],name='a')
        b = tf.constant([1.0,2.0,3.0],shape=[3],name='b')
    with tf.device('/gpu:1'):
        c = a+b

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))

def mytest_keras_gpu():
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(32, activation='relu',input_dim=100))
    model.add(Dense(2, activation='sigmoid'))
    model.summary()

    import numpy as np
    data = np.random.random((1000, 100))
    model.predict(data)


if __name__ == '__main__':
    mytest_flag = 0

    if mytest_flag:
        mytest_tf_gpu()
    else:
        mytest_keras_gpu()
