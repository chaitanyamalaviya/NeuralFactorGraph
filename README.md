# Neural Factor Graph Models for Cross-lingual Morphological Tagging

Morphological analysis involves predicting the syntactic traits of a word (e.g. {POS: Noun, Case: Acc, Gender: Fem}). Previous work in morphological tagging improves performance for low-resource languages (LRLs) through cross-lingual training with a high-resource language (HRL) from the same family, but is limited by the strict---often false---assumption that tag sets exactly overlap between the HRL and LRL. In this paper we propose a method for cross-lingual morphological tagging that aims to improve information sharing between languages by relaxing this assumption. The proposed model uses factorial conditional random fields with neural network potentials, making it possible to (1) utilize the expressive power of neural network representations to smooth over superficial differences in the surface forms, (2) model pairwise and transitive relationships between tags, and (3) accurately generate tag sets that are unseen or rare in the training data. Experiments on four languages from the Universal Dependencies Treebank demonstrate superior tagging accuracies over existing cross-lingual approaches.

### Prerequisites

CoNLL-U Parser (https://github.com/EmilStenstrom/conllu) :  ```pip install conllu```

PyTorch, version 0.3.0

### Usage

To run the baseline tagger for a language pair Danish/Swedish,

```
python baselineTagger.py --gpu --langs da/sv --tgt_size 1000
```

To run the Neural Factor Graph Model, 

```
python traincrf.py --gpu --langs da/sv --tgt_size 1000

```

The transitions and pairwise factors can be turned off with the `--no_transitions` and `--no_pairwise` arguments.

You can run evaluation with the argument `--test` and visualize the learnt parameter matrices with the `--visualize` argument.

### Data

The Universal Dependency Treebanks can be obtained from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2515.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Citation

```
TBA
```
