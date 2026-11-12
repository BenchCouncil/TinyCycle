# Efficiency Improvement of iTransformer

Supposing the input multivariate time series has a shape of $T \times N$. The vanilla attention module has a complexity of $\mathcal{O}(L^2)$, where $L$ is the number of tokens. 

* In Transformer, we have $L=T$ because of the manner of time points as tokens. 
* In iTransformer, we have $L=N$ because of the manner of variates as tokens. 

## Benefit from Efficient Attention

Since the attention mechanism is applied on the variate dimension in the inverted structure, efficient attention with reduced complexity essentially addresses the problem of numerous variates, which is ubiquitous in real-world applications.

We currently try out the linear complexity attention from [Flowformer](https://github.com/thuml/Flowformer), and the hardware-accelerated attention mechanism from [FlashAttention](https://github.com/shreyansh26/FlashAttention-PyTorch).

### Scripts 
We provide the iTransformers with the FlashAttention module:

```
# iTransformer on the Traffic Dataset with hardware-friendly FlashAttention for acceleration

bash ./scripts/model_efficiency/Traffic/iFlashTransformer.sh
```


## Efficient Training Strategy
With the input flexibility of attention, the token number can vary from training to inference, **our model is the first one to be capable of training on arbitrary numbers of series**. We propose a novel training strategy for high-dimensional multivariate series by taking advantage of the [variate generation capability](../variate_generalization/README.md). 

Concretely, we randomly choose part of the variates in each batch and only train the model with selected variates. Since the number of variate channels is flexible because of our inverting, the model can predict all the variates for predictions.
