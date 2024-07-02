# LLM-RCDA
The code implementation and data for [A **R**ationale-centric **C**ounterfactual **D**ata **A**ugmentation Method for Cross-Document Event Coreference Resolution](https://arxiv.org/pdf/2404.01921) accepted by NAACL24 main. Such data augmentation method is called LLM-RCDA.

Our implementation is built based on  [Focus on what matters: Applying Discourse Coherence Theory to Cross Document Coreference](https://github.com/Helw150/event_entity_coref_ecb_plus) repository. Thanks for their awsome work.

## Overview
1. Data preparation
   -  Retrieving mention pairs from ECB+/GVC/FCC's training/dev/test corpus.
   -  Interacting with LLMs.
   -  Generating CAD-enhanced dataset.

2. Training and evaluation    
   -  Zero-shot coreference evaluation results of LLMs with the Doc-Template prompt on ECB+.
   -  Zero-shot/Few-shot with CoT pairwise compairsion of LLMs on ECB+ test mention pairs.
   -  CAD quality evaluation.
   -  Fine-tune the coreference classifer with a Roberta-Large backbone.
   -  In-domain coreference evaluation on ECB+/FCC/GVC.
   -  Ablation study results on ECB+.
   -  Out-of-domain robustness test.   

## Environment
The codebase is tested with Python version 3.10.13. We recommend using [Conda](https://docs.anaconda.com/anaconda/) for managing your Python environment: 
```
conda create -n LLM-RCDA python=3.10.13
conda activate LLM-RCDA
```

Then install all the dependency packages by:

```
pip install -r requirements.txt
```

For the usage of spacy, the following command could be helpful:
```
python -m spacy download en_core_web_sm
```

## Train & evaluate the coreference classifier
The coreference classifer is built upon a transformer cross-encoder with Roberta-Large backbone. The code for training/evaluating the classifer is in [src/all_models/crossencoder_trainer.py](https://github.com/Danield21/Rationale4CDECR/blob/main/src/all_models/crossencoder_trainer.py)

Following [the previous SOTA work](https://github.com/Helw150/event_entity_coref_ecb_plus), $B^3$ F1 evaluated in the dev set is used to select the best model during training, and we also follow their setup to construct the pairwise dataset from the training/dev/test corpus. For [main experiments](https://github.com/Danield21/Rationale4CDECR/tree/main/configs/main) on ECB+/FCC/GVC, we retrieve the nearest 15 (K=15) and
5 (K=5) mention pairs for training and inference
in main experiments on three benchmarks. For the
[ablation study](https://github.com/Danield21/Rationale4CDECR/tree/main/configs/ablation_study), [augmentation ratio analysis](https://github.com/Danield21/Rationale4CDECR/tree/main/configs/AD_ratio), [the out-of-domain generalization test](https://github.com/Danield21/Rationale4CDECR/tree/main/configs/ood_test), we retrieve 5 (K=5) mention pairs for both training and inference. The processed pairwise dataset are presented in [retrieved_data](https://github.com/Danield21/Rationale4CDECR/tree/main/retrieved_data), and the constuction process of these datasets is introduced in [data_preparation](https://github.com/Danield21/Rationale4CDECR/tree/main/data_preparation).

Considering a trade-off between the training time
and the increasing amount of augmented data, we
only add two CAD for each original data from the
top 5 nearest pairwise data in the training set, and keep the others unchanged. After data augmentation, we receive 68.2K, 35.8K and 97.3K mention pairs to train the cross-encoder on ECB+, FCC and GVC respectively.

To train the model, please run the meta bash file: 
```
title=main # ablation_study or AD_ratio 
bash run_sh/${title}/train_crossencoder.sh
```
It will save the checkpoint of the best model, training logs, and coreferential links in the dev set to the correspounding `best_model' folder in outputs.

To evaluate the model on the test set, please run the meta bash file:
```
title=main # ablation_study or AD_ratio or ood_test
bash run_sh/${title}/eval_crossencoder.sh
```
It will save evaluation logs, coreferential links, gold_map and model_map in the test set to the correspounding `best_model' folder in outputs. The $B^3$ results are presented in the file crossencoder.eval.log

For other coreference metrics evaluation, such as $MUC$, $CEAF_e$, $LEA$ and $CoNLL$. Please see [conll_eval](https://github.com/Danield21/Rationale4CDECR/tree/main/conll_eval).



