import tensorflow as tf


class CoSentLoss(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        y_true = y_true[::2]
        norms = tf.math.reduce_sum(y_pred ** 2, axis = 1, keepdims=True) ** 0.5
        y_pred = y_pred / norms
        # y_pred = tf.math.l2_normalize(y_pred, axis=1, epsilon=0.5)
        y_pred = tf.math.reduce_sum(y_pred[::2] * y_pred[1::2], axis=1) * 20
        y_pred = y_pred[:, None] - y_pred[None, :]
        y_true = y_true[:, None] < y_true[None, :]
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = y_pred - (1 - y_true) * 1e12
        return tf.math.reduce_logsumexp(tf.concat([[0], tf.reshape(y_pred, -1)], 0))
