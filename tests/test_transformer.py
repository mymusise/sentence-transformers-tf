import unittest
import numpy as np
import torch
from sentence_transformers_tf.models import TFTransformer
from sentence_transformers.models import Transformer


class TFSentenceTransformerTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_encode(self):
        sentences = ["test sentence", "sentence test", "sentence for test"]
        model_id = "sentence-transformers/stsb-xlm-r-multilingual"
        model = Transformer(model_id)
        ids = model.tokenize(sentences)
        output = model(ids)

        tfmodel = TFTransformer(model_id)
        ids = tfmodel.tokenize(sentences)
        tfoutput = tfmodel(ids)

        diff = output['token_embeddings'].detach().numpy() - tfoutput['token_embeddings'].numpy()
        print(np.mean(abs(diff), axis=-1))

        assert np.mean(abs(diff)) < 1e-3

