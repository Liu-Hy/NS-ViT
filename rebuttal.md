# Rebuttal for nullspace paper
## Response to reviewer EpvQ
We sincerely appreciate your kind comments and your positive assessment. 

> ... In this regard, the authors are encouraged to furnish more elaborate details concerning the optimization-based approach employed.

$${\color{green}\text{I think we should add some details to make sure the reviewer does not back down on the scores. Same for the point below}}$$

> A more comprehensive explanation on the limitations and efficiency of this method in identifying the nullspace within the non-linear part would strengthen the paper.

## Response to reviewer fduk
We sincerely appreciate your kind comments and your positive assessment.

## Response to reviewer zvEG
We sincerely appreciate your kind comments and your insightful suggestions. Regarding the weaknesses, we hope our point-to-point response can address your concerns.\

>Invariance: The paper defines nullspace in a way that corresponds to the concept of "invariance," which has been extensively studied for many years ...

It is true that robustness is often defined as certain kind of invariance (such as translation, rotation or scaling) to the input, which has been widely studied in previous work. 
These invariances are decided based on the data/task and not the employed neural network. For example, image classification requires scale/translation/rotation invariance, however the same
cannot be extended to the task of say sentence classificaiton. In our paper, we do not focus on a specific type of invariance related to some human-understandable aspect of robustness. 
Instead, we study invariance as an inherent property of the vision transformer model, and empirically show its
connection with both adversarial and OOD robustness. 

We believe this notion of invariance is relatively under-studied in prior work. 
In linear algebra, vectors in certain directions are mapped to zero, and as a consequence, it does not change the model's output when added to any
input. Similarly, for a non-linear self-attention layer, we find that vectors along certain directions almost do not change the model output 
when added to the input. $\color{green}\text{I don't think we can say this, since it would imply sampling vectors along this direction also yield 0 for the non-linear layer.}$
Given the similar algebraic properties and implications to robustness, we think it is natural to extend the notion of
nullspace to nonlinear functions. That being said, we agree that a more explicit 
discussion on the concept of invariance, and the motivation of our definition of non-linear nullspace would improve the readability of 
our paper.


>Lack of comparison with previous work: The experiments conducted in the paper demonstrate that fine-tuning ViTs with nullspace noise improves robustness. However, the paper does not compare this method against existing approaches in the literature ...

We are sorry for the confusion. In Table 1, we compared the performance of our method with ViT-B against
several previous methods for robust training. However, this is not a fair comparison, because we fine-tuned the model with
very few steps due to the limited computation budget, it is not fair to compare it with
methods training ViT-B for hundreds of epochs. It still showed consistent improvement in different settings including
the current SOTA on ViT-B, which showed the effectiveness of our methods. Comparing our method with adversarial training would be very interesting. 

>To what extent does fine-tuning on null-space noise enhance adversarial robustness compared to fine-tuning on adversarial attacks?

## Response to reviewer uUyw
>The whole analysis of the null space seems totally unnecessary to me: why not simply say "a noise with -bounded-error in the output may improve the robustness"?

>The intuition behind adversarial training with "-approximate noise" and adversarial robustness is unclear.

>The authors should include vanilla adversarial training and compare against "ViT-S + NS".

>No ablation study over the choice of $\epsilon$

>The accuracy differences seem more or less like random noises. Without a confidence bound ...

>The whole story is not well organized - nobody cares about the existence of the null space unless it is shown ...

>I find Figure 2 irrelevant: the training dynamics does not show much interesting.

>I believe a much clearer intuition, a set of more robust results and a major rewriting of the paper is necessary
## Response to reviewer VJsT

