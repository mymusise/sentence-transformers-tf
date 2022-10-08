import unittest
import numpy as np
import torch
from sentence_transformers_tf.models import Pooling
from sentence_transformers.models import Pooling as STPooling


class PoolingTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_mean_pooling(self):
        pool = Pooling()
        st_pool = STPooling(0)

        num_sentences = 3
        len_sentences = 10
        dim_emebdding = 240
        embeddings = np.random.random((num_sentences, len_sentences, dim_emebdding))
        masks = np.ones((num_sentences, len_sentences))

        features = {'token_embeddings': embeddings, 'attention_mask': masks}
        features_pt = {'token_embeddings': torch.from_numpy(embeddings), 'attention_mask': torch.from_numpy(masks)}
        output = Pooling()(features)['sentence_embedding']
        st_output = STPooling(240)(features_pt)['sentence_embedding']

        for o1, o2 in zip(output[0][:10], st_output[0][:10]):
            assert abs(o1 - o2) <= 0.001
