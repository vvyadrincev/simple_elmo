# Simple ELMo
Minimal Python code to work with pre-trained ELMo models in TensorFlow.

Heavily based on https://github.com/allenai/bilm-tf

# Usage example

`python3 get_elmo_vectors.py -i test.txt -e ~/PATH_TO_ELMO/`

`PATH_TO_ELMO` is a directory containing 3 files:
- `model.hdf5`, pre-trained ELMo weights in HDF5 format
- `options.json`, description of the model architecture in JSON
- `vocab.txt.gz`, one-word-per-line vocabulary of the most frequent words you would like to cache during inference

Use the `elmo_vectors` tensor for our downstream tasks. 
Its dimensions are: (number of sentences, the length of the longest sentence, ELMo dimensionality).