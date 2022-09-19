# T5Patches

T5Patches is a set of tools for fast and targeted editing of generative language
models built with T5X.

## Documentation

### Objective
Encoder-decoder language models, despite their overall high performance, often exhibit suboptimal or undesirable behaviors such as poor style or hallucinations. These behaviors limit our ability to deploy these models without significant restrictions on their coverage. In this work, we introduce and examine methods to correct these models in a small fine-tuning step using corrected and targeted training examples.

### Background
Large language models, in particular T5/MUM, are pre-trained on a large corpora of text with a masked language objective. Although this objective is not directly applicable to practical applications, it helps the model understand the structure of language and relationships among various entities.

In the following fine-tuning step, models are often trained using a smaller, but sizable, “silver” training dataset specific to a task. The silver dataset can be extracted from logs, or generated synthetically by other systems. In this fine-tuning step, the model learns a specific mapping from a given “input” text to an “output” text.

Both the pre-training data and the silver data are imperfect- they contain biases, offensive or insensitive language, or just incorrect information. We expect a model trained on these imperfect datasets to be suboptimal.

The problem of editing a trained language model has received attention in recent years [[1](https://arxiv.org/abs/2112.00791), [2](https://arxiv.org/abs/2012.11635), [3](https://arxiv.org/abs/1909.08593), [4](https://arxiv.org/abs/1908.04319)]. We propose an alternative set of methods that are uniquely advantageous given the needs of product teams at Google, namely corrective and targeted negative training. Our methods edit a language model by additionally finetuning on corrected or annotated examples from the model's own generations. 


### Overview
In typical fine-tuning, a model is presented with (input, output) pairs. Both the input and the output are a sequence of tokens extracted from text by a tokenizer. At each timestep in training, the model is given the input and the preceding sequence of tokens from the output. It is asked to produce a probability distribution over all possible output tokens given the input and the preceding sequence of tokens from the output.

The parameters of the model are updated to increase the probability of the next token in the output by decreasing the loss for each step:

$$ loss = -log P_\theta(y|y_{prev},x)$$

Where $x$, $y$, and $\theta$ respectively refer to the input, the output, and model parameters.

At decoding time, without access to labeled training examples, the model uses the input along with the previously selected output tokens to estimate the probability of each subsequent output token. In the case of greedy decoding (e.g. beam size of 1), the token with the highest probability is selected.

By modifying this loss function, or by supplying new (input, output) pairs, one can increase or decrease the probability of certain tokens given an input and a sequence of previous output tokens. In the following section we describe strategies to encourage or discourage certain generations using these techniques.

### Detailed Design
#### Corrective Training
Given a set of undesirable (input, output) pairs (outputs that are inappropriate or incorrect, given the input), we can collect ‘corrections’ to the output. Corrections can be from human raters, from a more powerful generative language model, or from rules. Further fine-tuning a model using a small number of corrections for a few training steps has shown to be effective in modifying characteristics of models. See the Results section for details.


#### Targeted Training with Modified Self-Distillation
In targeted training, we modify the model by directly changing the probability distribution over all possible output tokens. This allows us to  increase or decrease the probability of each output token, while minimizing the change to the probability of other tokens. We accomplish this using a modified loss function, the KL Divergence between the model’s distribution (P_{model}) and a modified version of the original model distribution (P_{modified}) for each conditional distribution:

$$ loss = D_{KL}(P_{modified}, P_{model}).$$

#### Use cases: Corrective vs. Targeted negative training
Collecting corrections can be relatively more expensive and cumbersome than marking individual tokens for targeted training. However, the application of corrections to a model is straightforward. Model training can be simply continued for a few extra steps with corrected examples.

In targeted training, we can specify token probabilities to push down. Note that which tokens are selected for pushing down can affect what sort of outputs are being discouraged. For example, a model developer might be interested in discouraging outputs that start with ‘what is.’ Pushing down the probability of ‘what’ as the first token will discourage all generations that might start with ‘what,’ including ‘what is,’ ‘what does,’ ‘what was,’ etc. Pushing down the probability of ‘is’ when the first output token is ‘what’ will discourage ‘what is’ outputs, but not other ‘what’ outputs.

When undesirable patterns can be expressed either as examples with specific undesired tokens or simple rules applied to specific tokens, targeted training can be a good choice for controlling generations.

#### Implementation Details
To test corrective and negative training within one unified framework, we developed a set of feature converters that take in data containing both “negative_targets” and “corrected_targets” entries and determine the targets to be used in training, alongside their associated weights. We also developed  new encoder-decoder models that utilize a different loss to handle outputs with both positive and negative weights.

#### How to use
For corrective training, once the corrective outputs are created, one can simply finetune on such outputs using the standard fine-tuning api. (We had tested other forms of setting weights to the corrections, but none seemed significantly better than typical finetuning to justify suggesting an alternative approach.)

For targeted negative training, create a dataset with inputs, negative_targets, and corrected_targets. The corrected_targets can simply be a copy of the negative_targets, except for the tokens one wishes to push down (for those tokens, the corrected_targets can simply be a dummy token or token sequence of the same length as the negative tokens of interest).

### Notes and Considerations
In our experiments with the T5/MUM Base model (600M parameters) we have been able to edit the model’s style with only a few hundred or thousand corrective or negative targeted examples. For small edits, we have been able to update a model in as little as ten steps with corrective training, or 10k steps for for targeted training--both orders of magnitude smaller than finetuning from scratch.


## Support

* Team: zlily@, aryatafvizi@, hqian@
