# Data preparation 
## 1. Capture the retrieved event mention pairs from the training/dev/test corpus.

- Train a bi-encoder for event embedding in the latent space.

- According the event embedding, we can retrieve the nearest-K neighbours for each event mention in the training/dev/test corpus to construct the pairwise dataset to train the cross-encoder which is a coreference scorer. 
  - Aligned with the baseline work of Held, we set K as 15(train)/5(dev)/5(test) for main experiments, and 5(train)/5(dev)/5(test) for other experiments. This operation is done with the function `nn_generate_pairs' in the file src/all_models/nearest_neighbor.py and the retrieved data are stored in the dir retrieved_pairs.

## 2. Construct augmented data
  - Based on alg. 1 in the paper, we interact with LLMs to generate materials for data augmentation.
  -  Consolidate generated materials to obtain augmented data.