# TinyCycle-plug on PEMS

See the [TinyCycle-plug instructions](README_TinyCycle_PEMS.md) for automated validation search and test evaluation.

This package uses the upstream [THUML iTransformer implementation](https://github.com/thuml/iTransformer). The iTransformer paper and its author list are available on [arXiv](https://arxiv.org/abs/2310.06625). See the upstream [MIT license](LICENSE).

## Upstream iTransformer examples

1. Install Pytorch and the necessary dependencies.

```
pip install -r requirements.txt
```

2. The datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Baidu Cloud](https://pan.baidu.com/s/11AWXg1Z6UwjHzmto4hesAA?pwd=9qjr).

3. Run a training and evaluation script.

```
# Multivariate forecasting with iTransformer
bash ./scripts/multivariate_forecasting/Traffic/iTransformer.sh

# Compare the performance of Transformer and iTransformer
bash ./scripts/boost_performance/Weather/iTransformer.sh

# Train the model with partial variates, and generalize to the unseen variates
bash ./scripts/variate_generalization/ECL/iTransformer.sh

# Test the performance on the enlarged lookback window
bash ./scripts/increasing_lookback/Traffic/iTransformer.sh

# Utilize FlashAttention for acceleration
bash ./scripts/model_efficiency/Traffic/iFlashTransformer.sh
```
