# Data preparation 
## 1. Download and preprocess corpus.
Our experiments are on three cross-document corpus: ECB+, FCC and GVC.
You are free to directly use our preprocessed data in the dir [../feature_sets](https://github.com/Danield21/Rationale4CDECR/tree/main/feature_sets). Also, you can download and preprocess these corpus on your own by:
```
bash ./data/download_dataset.sh 
``` 

## 2. Embed event by training a bi-encoder.
A bi-encoder can be trained for event embedding in a latent semantic space where coreferential events are close while non-coreferential ones are away from each other in the space.


## 3. Capture nearest-K mention pairs.   
According the event embedding, we can retrieve the nearest-K neighbours for each event mention in the training/dev/test corpus to construct the pairwise dataset to train the cross-encoder classifer as a coreference scorer. Aligned with [the baseline work](https://aclanthology.org/2021.emnlp-main.106.pdf), we set K as 15(train)/5(dev)/5(test) for main experiments, and 5(train)/5(dev)/5(test) for other experiments. Please refer to:
```

```



## 4. Construct augmented data
  - Based on alg. 1 in the paper, we interact with LLMs to generate materials for data augmentation.
  -  Consolidate generated materials to obtain augmented data.