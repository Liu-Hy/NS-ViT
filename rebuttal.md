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

Figure (2) highlights the training dynamics with change in the value of $\epsilon$. 

>The accuracy differences seem more or less like random noises. Without a confidence bound ...

>The whole story is not well organized - nobody cares about the existence of the null space unless it is shown ...

>I find Figure 2 irrelevant: the training dynamics does not show much interesting.

We have moved the figure to section 5 after we introduce the concept in equation 7.

>I believe a much clearer intuition, a set of more robust results and a major rewriting of the paper is necessary
## Response to reviewer VJsT

> ... The core of the idea in this paper is then a method to approximate the largest noise such that the model output (measure via the loss) is invariant that is not at all vision transformer specific nor theoretically grounded.

> The introduction is quite lengthy and involves several paragraph of generic discussions of vision transformers  ...

> Section 3: while it's interesting that the linear patch embedding allows us to describe the null space in terms of strict invariance, this analysis is no longer valid when we consider the entire model ...

Respectfully, we maintain that the study of linear patch embedding is important. This is because the nullspace of linear patch embedding is also the exact nullspace of the entire model as well.

> ... How does this method compare against training on random epsilon noise?

>  I suggest the authors tone down the claim “unexplored concept of nullspace…” ...

> Equation 6 missing the optimization goals and constraints, which come later in lengthy discussion ...

> typo ...

We have addressed them, and the corrections have been made. 

>  “fewer engineering hurdles” can you clarify what you mean by this?

We highlight ViTs are the foundation versions of the much complex architectures developed later on and in the corresponding paragraph, we attempt to justify our reasoning for only working with ViTs. The complexities introdcued by later models are usually in form of hybridisation with CNNs or alterations to the multi-head attention blocks. Given the immediate accessibility of the source codes, pre-trained weights and the wealth of available resources, there exists a compelling inclination to employ the Vision Transformer (ViT) models as opposed to any other much recent transformer based architecture.  

> Can you clarify what the statement of Proposition 1 Condition 2? Are you stating the column space of V is contained in the row space of QK^T for each attention head? And can you justify why this assumption holds?

> What’s the justification for Proposition 1. Condition 3?

> Does the optimization in equation 6 only search for a single element statisfying closeness under addition? If so how many vectors do you learn and what are their values?

> Figure 2 references epsilon on the x-axis, but epsilon is not defined in the optimization in equation 6.

We apologise for the confusion. $\epsilon$ is introduced later on in equation (7), consequently, we have shifted this ablation discussion to section 5.

> “as the epsilon criteria becomes smaller, the learnt noise becomes better and better” Is this conclusion based on the % of matching predictions? ...

This is based on both the metrics reported in the plot (1) matching predictions and (2) Mean-squared error of the confidence values of ground-truth categories.

> Our preliminary experiments indicate that there may exist a non-isomorphic space in the input space ...

> Equations 7 and 8 rehash the optimization you already present in Section 4.

> I’m having a hard time following this logic. If the procedure you describe identifies vectors in the approximate null space, then by definition ...

> I’m quite surprised ViT-S accuracy improves so dramatically for ImageNet A and Sketch ...

> This paper shows training with https://arxiv.org/abs/1808.08750 augmenting salt-and-pepper ...

> The application to model patenting is not clear to me at all ...

The model patenting references to the exact nullspace noise which is derived from the linear patch embedding layer of the ViTs. It is true that nullspace noise might not be unique to a particular instance of the model, however, by generating a large number of such noise samples we can confidently pursue any violations. Cases where exact nullspace is not a viable solution, we can employ approximated noise in a similar fashion of synthesizing many samples and performing verificaiton.

> I’m not sure what insight this work sheds in terms of how transformer models are robust to input perturbations ...

> In the proof of proposition 1, what what r_{m, k}?

> For the ViT-L 16x16, the computed nullspace dimension is 0 ...

