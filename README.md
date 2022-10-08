# Sentence-Transformers-TF: UnOfficial TensorFlow Implementation

# How to use

```python
from sentence_transformers_tf.TFSentenceTransformer import TFSentenceTransformer

tfsentent = TFSentenceTransformer("sentence-transformers/stsb-xlm-r-multilingual")

outs = tfsentent.encode(["hi, there", "tihs is a tensorflow implementation of the sbert"])
print(outs.shape)
```
