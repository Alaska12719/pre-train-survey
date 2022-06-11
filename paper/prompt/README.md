# Propmt Tuning

Unlike traditional supervised learning, which model input $x$and predict an output $y$ as $P(y|x)$, **prompt-based** learning is based on language models that model the probability of text directly. So the input $x$ is modified using a template into a textual string prompt $x^{'}$ that has some unfilled slots, and then the language model fill the information to obtain a final string $\hat{x}$.

**Why it is powerful?** It allows the language model to be pre-trained on massive amount of raw text, and by defining a new prompting function the model is able to perform *few-shot* or even *zero-shot* learning, adapting to new scenarios with few or no labled data.

Prompting predicts the highest-scoring $\hat{y}$ in 3 steps.

1. prompt addition
   a function $f_{prompt}(·)$ modify the input text $x$ into a prompt $x^{'}$, containing [z] slot and [x] slot. [x] slot is input text (question) and [z] slot is answer.
2. answer search
   search highest-scoring text $\hat{z}$ that maximaizes the score of the LM$P(x;\theta)$.
3. answer mapping
   go from highest-scoring answer $\hat{z}$ to the highest-scoring output $\hat{y}$.

Design Consideration for Prompting

1. Pre-trained Model Choice: calculate $P(x;\theta)$.
2. Prompt Engineering: choose which prompt we should use as $f_{prompt}(x)$.
3. Answer Engineering: design $ \mathcal{z}$ differently(the set of answer), possibly along with the mapping function.
4. Expanding the Paradigm: above is the simplest of the various underlying frameworks, there are ways to expand this underlying paradigm.
5. Prompt-based Training Strategies: train parameters either of the prompt, the LM, or both.
