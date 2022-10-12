# Sentence-Transformers-TF: 

## A unofficial TensorFlow Implementation of SBERT


<p align="center">
    <a href="https://colab.research.google.com/github/mymusise/sentence-transformers-tf/blob/main/examples/finetune.ipynb">
        <img alt="Build" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://travis-ci.org/mymusise/sentence-transformers-tf.svg?branch=master">
        <img alt="Build" src="https://travis-ci.org/mymusise/sentence-transformers-tf.svg?branch=master">
    </a>
    
</p>

# Inference

```python
from sentence_transformers_tf.TFSentenceTransformer import TFSentenceTransformer

tfsentent = TFSentenceTransformer("sentence-transformers/stsb-xlm-r-multilingual")

outs = tfsentent.encode(["hi, there", "tihs is a tensorflow implementation of the sbert"])
print(outs.shape)
```

# Finetune

Here is an example: [examples/finetune.ipynb](examples/finetune.ipynb)


# Reference

- Origin repository: [https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- Original Paper : [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)