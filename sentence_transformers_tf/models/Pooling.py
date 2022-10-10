import tensorflow as tf


class Pooling(tf.keras.layers.Layer):
    def mean_pooling(self, token_embeddings, attention_mask=None):
        if attention_mask is None:
            return tf.math.reduce_mean(token_embeddings, axis=1)
        input_mask_expanded = tf.cast(tf.broadcast_to(tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)), tf.float32)
        output = tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1) / tf.clip_by_value(tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max)
        return output

    def call(self, features):
        token_embeddings = features["token_embeddings"]
        attention_mask = features.get("attention_mask")

        output_vector = self.mean_pooling(token_embeddings, attention_mask)
        features.update({'sentence_embedding': output_vector})
        return features
