from tkinter import S
import tensorflow as tf
import logging
from typing import Optional, Iterable, Union, List
import numpy as np
from .models import TFTransformer, Pooling
from tqdm.autonotebook import trange


logger = logging.getLogger(__name__)


class TFSentenceTransformer(tf.keras.Model):
    """
    Loads or create a SentenceTransformer model, that can be used to map sentences / text to embeddings.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
    :param modules: This parameter can be used to create custom SentenceTransformer models from scratch.
    :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
    :param cache_folder: Path to store models. Can be also set by SENTENCE_TRANSFORMERS_HOME enviroment variable.
    :param use_auth_token: HuggingFace authentication token to download private models.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        modules: Optional[Iterable[tf.keras.layers.Layer]] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        use_auth_token: Union[bool, str, None] = None,
        *inputs,
        **kwargs,
    ):
        super().__init__(*inputs, **kwargs)
        self._model_card_vars = {}
        self._model_card_text = None
        self._model_config = {}

        if not modules:
            self.model = TFTransformer(model_name_or_path)
            self.pooling_model = Pooling()
        else:
            self.model = modules[0]
            self.pooling_model = modules[1]

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[tf.Tensor], np.ndarray, tf.Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.model.tokenize(sentences_batch)
            out_features = self(features)
            embeddings = out_features[output_value]
            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = tf.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        return all_embeddings

    def call(self, features, **kwargs):
        if isinstance(features, tf.Tensor):
            features = {'input_ids': features}
        out_features = self.model(features, **kwargs)
        out_features = self.pooling_model(out_features)
        return out_features

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs, labels = data
        # return super().train_step(inputs)
        sentence1 = inputs.get('input_ids')
        sentence2 = inputs.get('target_ids')

        with tf.GradientTape() as tape:
            features1 = self(sentence1, training=True)  # Forward pass
            features2 = self(sentence2, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss((features1['sentence_embedding'], features2['sentence_embedding']), labels)

            sim = -tf.keras.losses.cosine_similarity(features1['sentence_embedding'], features2['sentence_embedding'])
            labels = tf.cast(labels, dtype=sim.dtype)
            loss = tf.reduce_mean(tf.math.square(sim - labels), axis=-1)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(labels, (features1, features2))
        # Return a dict mapping metric names to current value
        return {'loss_value': loss}
        # return {m.name: m.result() for m in self.metrics}

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

