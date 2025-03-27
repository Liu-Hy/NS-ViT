# Rebuttal for nullspace paper
## Response to reviewer EpvQ
We sincerely appreciate your kind comments and your positive assessment. We will carefully consider your suggestions and
add more thorough discussions in future revision on the proposed optimization method for the nullspace noise.

## Response to reviewer fduk
We sincerely appreciate your kind comments and your positive assessment.

## Response to reviewer zvEG
We sincerely appreciate your kind comments and your insightful suggestions. Regarding the weaknesses, we hope our point-to-point response can address your concerns.\
**W1. Regarding the concept of nullspace and invariance** \
It is true that robustness is often defined as certain kind of invariance, which has been widely studied in previous work. 
However, in our paper, we do not focus on a specific type of invariance related to some human-understandable aspects of robustness. 
Instead, we study invariance as an inherent property of the vision transformer model, and empirically show its
connection with both adversarial and OOD robustness. We believe this notion of invariance is relatively under-studied in prior work.
In linear algebra, vectors in certain directions are mapped to zero, and
as a consequence, it does not change the model's output when added to any
input. Similarly, for a non-linear self-attention layer, we find that vectors along certain directions almost do not change the model output 
when added to the input. Given the similar algebraic properties and implications to robustness, we think it is natural to extend the notion of
nullspace to nonlinear functions. That being said, we agree that a more explicit 
discussion on the concept of invariance, and the motivation of our definition of non-linear nullspace would improve the readability of 
our paper.


**W2. Regarding comparison with previous work**
We are sorry for the confusion. 