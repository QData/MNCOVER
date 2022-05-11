# MNCOVER

This repository contains code for white box testing of NLP models as described in the following paper:  
>[White-box Testing of NLP models with Mask Neuron Coverage](https://arxiv.org/abs/2004.11494)  
> https://arxiv.org/abs/2205.05050
> Findings of Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2022

## Trained Masks and Initialization Files
All trained masks and initialization files can be found [here](https://drive.google.com/drive/folders/1u1a4gsUU4KGPQfUv_d1-MoaeL1UFEqo5?usp=sharing). 

## Initializing Coverage 
```
python initialize_coverage.py --seed 1 --bins-word 10 --bins-attention 10   --max-seq-len 128 --batch-size 128 --alpha 1.0 --test-name "change names"  --suite sentiment --subset 1500 --base-model roberta-base --save-dir results/
```

## Filtering and Calculating Failure Rates 
```
python calculate_coverage.py --seed 1 --bins-word 10 --bins-attention 10   --max-seq-len 128 --batch-size 128 --alpha 1.0 --test-name "change names"  --suite sentiment --subset 1500 --base-model roberta-base --save-dir results/
```

