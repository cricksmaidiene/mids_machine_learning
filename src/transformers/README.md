# Transformers

Transformers are a sequence to sequence (*seq2seq*) based neural-network architecture, and a successor to LSTMs. They are faster to train, and have a better contextual understanding of language compared to LSTMs (even bi-directional LSTMs do not capture meaning in the best way). As such, transformers are great for use-cases that involve natural language "understanding", or where meaning is learnt by the model.

Two main classes of transformer architectures are:
1. `BERT` (Bi-Directional Encoder Representations from Transformers)
2. `GPT` (Generative Pre-Trained Transformer)

These classes are composed from the building blocks of transformers - the `encoder` and `decoder` layers. BERT involves a stack of encoder layers, and GPT is a stack of decoder layers. 

Transformers were born out of the seminal paper "Attention is all you need", which outlines that models can focus their attention on specific parts of a sentence or language to derive meaning. Their key innovation, the self-attention mechanism, allows them to consider the entire input sequence simultaneously, leading to improvements in processing speed and contextual understanding

## BERT

BERT models can be `pretrained` and `fine-tuned`. Pretraining means that a basic language understanding is provided to the model, and it learns some elements of language and stores this in its "memory". Fine-tuning means taking a `pretrained` model and providing it with a large set of labeled data to solve a specific problem. Fine-tuning a pretrained model is often refered to as `Transfer Learning`. 



