import unittest
import numpy as np
import torch
from sentence_transformers_tf import TFSentenceTransformer
from sentence_transformers import SentenceTransformer


class TFSentenceTransformerTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_encode(self):
        sentences = ["test sentence", "sentence test", "sentence for test"]
        model_id = "stsb-xlm-r-multilingual"
        model = SentenceTransformer(model_id)
        output = model.encode(sentences)

        tfmodel = TFSentenceTransformer(f"sentence-transformers/{model_id}")
        tfoutput = tfmodel.encode(sentences)

        diff = output - tfoutput
        print(np.mean(abs(diff), axis=-1))

        assert np.mean(abs(diff)) < 1e-3
