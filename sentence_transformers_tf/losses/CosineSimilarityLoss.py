import tensorflow as tf


class CosineSimilarityLoss(tf.keras.losses.Loss):

    def call(self, labels, sentence_features):
        features1, features2 = tf.split(sentence_features, 2)
        sim = -tf.keras.losses.cosine_similarity(features1, features2)
        labels = tf.cast(labels, dtype=sim.dtype)
        loss = tf.reduce_mean(tf.math.square(sim - labels), axis=-1)
        return loss

