# Data preparation 
## 1. Download and preprocess corpus.
Our experiments are on three cross-document corpus: ECB+, FCC and GVC.
You are free to directly use our preprocessed data in the dir [../feature_sets](https://github.com/Danield21/Rationale4CDECR/tree/main/feature_sets). Also, you can download and preprocess these corpus on your own by:
```
bash ../run_sh/data_prepare/download_dataset.sh 
``` 

## 2. Embed event by training a bi-encoder.
A bi-encoder can be trained for event embedding in a latent semantic space where coreferential events are close while non-coreferential ones are away from each other in the space. If you want to customize your own bi-encoder, please refer to the code in [../src/train_candidate_generation.py](https://github.com/Danield21/Rationale4CDECR/blob/main/src/all_models/train_candidate_generation.py) and the [config for ECB+](https://github.com/Danield21/Rationale4CDECR/blob/main/configs/bi_encoder/ecb/candidate_generator_train_config.json). 

Trained bi-encoder models can be downloaded by:
https://drive.google.com/file/d/15G_AxNPDvy90eKXookS7EWrdjr8VhF6E/view?usp=drive_link
Please un-zip the file to [../output](https://github.com/Danield21/Rationale4CDECR/tree/main/outputs) 

## 3. Capture nearest-K mention pairs.   
According the event embedding, we can retrieve the nearest-K neighbours for each event mention in the training/dev/test corpus to construct the pairwise dataset to train the cross-encoder classifer as a coreference scorer. Aligned with [the baseline work](https://aclanthology.org/2021.emnlp-main.106.pdf), we set K as 15(train)/5(dev)/5(test) for main experiments, and 5(train)/5(dev)/5(test) for other experiments. Please refer to:
```
bash ../run_sh/data_prepare/retrieve_mention_pairs.sh
```
You can download the pairwise data by:
https://drive.google.com/file/d/1TSDVho-CXByavWox4dT8cxWO-gn6fFpR/view?usp=drive_link
Please un-zip the file to the project root dir. 


## 4. Interact with the LLM
Based on alg. 1 in the paper, we interact with LLMs to generate materials for data augmentation. Accourding to the coreference of the original mention pair $MP=(m_1, m_2)$, we design correspounding counterfactual augmented data (CAD) to enhance the performance of  cross-encoder classifier.
- If mentions in $MP$ are coreferential, we let LLM generate some non-coreferential event mentions to $m_1$.
- If mentions in $MP$ are non-coreferential, we let LLM generate some coreferential event mentions to $m_1$. Also, we paraphrase the prefix and suffix discourse of $m_1$.   



## 5. Construct augmented data
Consolidate generated materials to obtain augmented data.