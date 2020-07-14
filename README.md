# An EM Approach to Non-autoregressive Conditional Sequence Generation

Here are the code for [the ICML paper](https://arxiv.org/abs/2006.16378). We provide the training scripts, as well as the distilled data used in our paper.
The code is adapted from the code of [BERT](https://github.com/google-research/bert).

The model is trained on TPU and evaluated on GPU.

## Prerequisites

* Python = 2.7.13
* TensorFlow = 1.14.0
* numpy = 1.16.5
* [subword-nmt](https://github.com/rsennrich/subword-nmt)
* [mosesdecoder](https://github.com/moses-smt/mosesdecoder)

## Usage

We use the [fairseq scripts](https://github.com/pytorch/fairseq/tree/master/examples/translation) to obtain and pre-process the data.

The training/eval/scoring scripts can be found in the `scripts` folder.

The `Transformer-TPU` folder contains the Python code for our model.

The `configs` folder contains three Transformer settings



## Citation
```
@incollection{icml2020_2711,
 abstract = {Autoregressive (AR) models have been the dominating approach to conditional sequence generation, but are suffering from the issue of high inference latency.  Non-autoregressive (NAR) models have been recently proposed to reduce the latency by generating all output tokens in parallel but could only achieve inferior accuracy compared to their autoregressive counterparts, primarily due to a difficulty in dealing with the multi-modality in sequence generation.  This paper proposes a new approach that jointly optimizes both AR and NAR models in a unified Expectation-Maximization (EM) framework. In the E-step, an AR model learns to approximate the regularized posterior of the NAR model. In the M-step, the NAR model is updated on the new posterior and selects the training examples for the next AR model. This iterative process can effectively guide the system to remove the multi-modality in the output sequences and remedy the multi-modality problem. To our knowledge, this is the first EM approach to NAR sequence generation. We evaluate our method on the task of machine translation. Experimental results on benchmark data sets show that the proposed approach achieves competitive, if not better, performance with existing NAR models and significantly reduces the inference latency.},
 author = {Sun, Zhiqing and Yang, Yiming},
 booktitle = {Proceedings of Machine Learning and Systems 2020},
 pages = {4716--4725},
 title = {An EM Approach to Non-autoregressive Conditional Sequence Generation},
 year = {2020}
}
```