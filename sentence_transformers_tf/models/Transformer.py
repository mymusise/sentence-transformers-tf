import tensorflow as tf
from typing import Optional, Dict, Union, List, Tuple
from transformers import TFAutoModel, AutoTokenizer, AutoConfig, T5Config, MT5Config


class TFTransformer(tf.keras.layers.Layer):
    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: Optional[int] = None,
        model_args: Dict = {},
        cache_dir: Optional[str] = None,
        tokenizer_args: Dict = {},
        do_lower_case: bool = False,
        tokenizer_name_or_path: str = None,
    ):
        super(TFTransformer, self).__init__()

        self.do_lower_case = do_lower_case

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self._load_model(model_name_or_path, config, cache_dir, **model_args)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path, cache_dir=cache_dir, **tokenizer_args)
        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation="longest_first", return_tensors="tf", max_length=self.max_seq_length))
        return output

    def _load_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
        elif isinstance(config, MT5Config):
            self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
        else:
            self.auto_model = TFAutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, from_pt=True, **model_args)

    def _load_t5_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the encoder model from T5"""
        from transformers import TFT5EncoderModel

        TFT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = TFT5EncoderModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, **model_args)

    def call(self, features, **kwargs):
        trans_features = {"input_ids": features["input_ids"]}
        for key in ["attention_mask", "token_type_ids"]:
            if key in features:
                trans_features[key] = features[key]

        output_states = self.auto_model(**trans_features, return_dict=False, **kwargs)
        output_tokens = output_states[0]

        features.update({"token_embeddings": output_tokens})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size
