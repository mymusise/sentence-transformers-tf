import tensorflow as tf


class CosineSimilarityLoss(tf.keras.losses.Loss):

    def __init__(self, *args, loss_fn=tf.keras.losses.MeanSquaredError, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn()

    def call(self, sentence_features, labels):
        sim = -tf.keras.losses.cosine_similarity(sentence_features[0], sentence_features[1])
        labels = tf.cast(labels, dtype=sim.dtype)
        return tf.reduce_mean(tf.math.square(sim - labels), axis=-1)

