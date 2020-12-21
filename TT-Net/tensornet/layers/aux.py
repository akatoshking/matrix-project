import tensorflow as tf

# Define a variable node in computation graph, see definition of get_variable
def get_var_wrap(name,
                 shape,
                 initializer,
                 regularizer,
                 trainable,
                 cpu_variable):
    if cpu_variable:
        with tf.device('/cpu:0'):
            return tf.get_variable(name,
                                   shape=shape,
                                   initializer=initializer,
                                   regularizer=regularizer,
                                   trainable=trainable)
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           regularizer=regularizer,
                           trainable=trainable)
