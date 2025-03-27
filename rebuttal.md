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

We compared our method with fine-tuning using two PGD adversarial training methods, Madry [1] and TRADES [2] on the ViT-S model. Below show the results.

|  method   | clean | FGSM  | DamageNet |   A   | C (⭣) |  V2   |   R   | Sketch | Stylized |
|:---------:|:-----:|:-----:|:---------:|:-----:|:-----:|:-----:|:-----:|:------:|:--------:|
| Nullspace | 77.47 | 25.95 |   32.43   | 20.77 | 55.98 | 66.5  | 41.61 | 25.67  |  16.02   |
|   Madry   | 70.53 | 39.37 |   49.91   | 9.37  | 81.74 | 58.88 | 39.04 | 21.36  |  10.76   |
|  TRADES   | 74.02 | 38.85 |   36.28   | 16.53 | 73.11 | 63.37 | 40.86 | 26.43  |  13.22   |

Although these two methods are effective in improving the adversarial robustness of the model,
they reduce the clean accuracy, and show much weaker performance on the OOD robustness in general.

References:\
[1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models 
resistant to adversarial attacks. ICLR, 2018.\
[2] Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, and Michael I. Jordan. Theoretically principled 
trade-off between robustness and accuracy. In ICML, 2019.


## Response to reviewer uUyw
>The whole analysis of the null space seems totally unnecessary to me: why not simply say "a noise with -bounded-error in the output may improve the robustness"?

In this paper, we study robustness from the inherent property of vision transformers, so the starting point is to analyze 
those properties. We established the existence of nullspace in the patch embedding layer, extended the concept of nullspace
to the non-linear function, and showed the existence of non-linear approximate nullspace. Conceptually, the existence of nullspace directly
implies the model's tolerance to certain (potentially large) perturbations, which indicates robustness. Subsequently, our
experiments with the noise augmentation method verified the connection
between nullspace and robustness. We hope our analysis brings a new perspective to the robustness of vision transformers.


>The intuition behind adversarial training with "-approximate noise" and adversarial robustness is unclear.

Our intuition is that, a transformer model should be invariant to large perturbations as long as they do not
change the semantic information of the image. We know that such large perturbations exist, such as the Gaussian noise. 
If a vision transformer model is robust, it should possess this kind of invariance. We do not focus on a specific kind of
perturbation, but on whether there exist some large perturbations that vision transformers are invariant to. The nullspace
of the linear embedding layer is one source of such invariance, and the approximate nullspace is another. By noise augmentation,
we simultaneously enlarged the approximate nullspace and robust accuracies on multiple datasets, validating our intuition. 

>The authors should include vanilla adversarial training and compare against "ViT-S + NS".

We compared our method with fine-tuning using two PGD adversarial training methods, Madry [1] and TRADES [2] on the 
ViT-S model. Below show the results.

|  method   | clean | FGSM  | DamageNet |   A   | C (⭣) |  V2   |   R   | Sketch | Stylized |
|:---------:|:-----:|:-----:|:---------:|:-----:|:-----:|:-----:|:-----:|:------:|:--------:|
| Nullspace | 77.47 | 25.95 |   32.43   | 20.77 | 55.98 | 66.5  | 41.61 | 25.67  |  16.02   |
|   Madry   | 70.53 | 39.37 |   49.91   | 9.37  | 81.74 | 58.88 | 39.04 | 21.36  |  10.76   |
|  TRADES   | 74.02 | 38.85 |   36.28   | 16.53 | 73.11 | 63.37 | 40.86 | 26.43  |  13.22   |

Although these two methods are effective in improving the adversarial robustness of the model,
they reduce the clean accuracy, and show mush weaker performance on the OOD robustness in general.

References:\
[1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models 
resistant to adversarial attacks. ICLR, 2018.\
[2] Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, and Michael I. Jordan. Theoretically principled 
trade-off between robustness and accuracy. In ICML, 2019.

>No ablation study over the choice of $\epsilon$

Figure (2) highlights the training dynamics with change in the value of $\epsilon$. Moreover, we report the robustness evaluation with different choices of $\epsilon$ below

| $\epsilon$ | clean | FGSM  | DamageNet |   A   | C (⭣) |  V2   |   R   | Sketch | Stylized |
|:----------:|:-----:|:-----:|:---------:|:-----:|:-----:|:-----:|:-----:|:------:|:--------:|
|    0.01    | 77.44 | 26.04 |   33.65   | 20.45 | 56.26 | 66.47 | 41.4  | 23.34  |  15.85   |
 |    0.03    | 77.47 | 25.95 |   32.43   | 20.77 | 55.98 | 66.5  | 41.61 | 25.67  |  16.02   |
|    0.1     | 77.06 | 25.38 |   33.09   | 20.16 | 56.41 | 66.47 | 40.42 | 22.66  |  15.78   |

>The accuracy differences seem more or less like random noises. Without a confidence bound ...

The ImageNet-C dataset contains over 3 million test images, requiring around 10 hours to evaluate on a 3090 GPU. The computation
workload is prohibitively high. To our knowledge, almost all the previous methods on robust ViT training reported results on a single run. 
We hope the scale and variety of the test datasets, and the ablation studies are enough to demonstrate the effectiveness of our method.

>The whole story is not well organized - nobody cares about the existence of the null space unless it is shown ...

We believe we have justified the need for studying the nullspace in our previous response. The existence of nullspace in vision transformers directly implies its robustness
to certain perturbations. This may shed light on its robust behaviors and provide a way to improve robustness in both adversarial and OOD settings. We will
refine our writing to make our ideas more explicit.

>I find Figure 2 irrelevant: the training dynamics does not show much interesting.

Figure 2 shows the result of our preliminary experiment of nullspace training. We used dashed line to show the model's performance after permuting the elements of the noise vector (Line 202). The permutation preserves the norm of the noise but randomly resets its direction, which results in strong influence to the model's predictions. This indicates that by gradient descent,
we can learn noise vectors which are large enough to break the model if towards arbitrary direction,
but whose direction is learned to produce little influence to the model.

>I believe a much clearer intuition, a set of more robust results and a major rewriting of the paper is necessary

For the intuition, let us briefly summarize our response to the weakness part. In this paper, we study robustness from the inherent property of vision transformers. We 
discuss and empirically verify the existence of nullspace from the patch embedding layer, and
extend the concept of nullspace to the non-linear function preserving the closeness under additivity.
The existence of (approximate) nullspace directly imply the model's invariance to certain perturbations. The connection between nullspace
and robustness is verified by our experiment results, which also serve as a method to enhance robustness
of vision transformers.

In terms of experiment results, we evaluated our method on different models and multiple datasets, and tracked the change of the nullspace noise magnitude and 
different aspects of performance along the optimization process. Result shows that our method is robust to different models and different epsilon values. 

As for the writing, we agree that rewriting some paragraphs with more detailed explanation on our intuition would improve the readability of our paper. 
We appreciate your suggestions.

## Response to reviewer VJsT
We sincerely appreciate your kind comments and your insightful suggestions.
Regarding the weaknesses, we hope our point-to-point response can address your concerns.\

> ... The core of the idea in this paper is then a method to approximate the largest noise such that the model output (measure via the loss) is invariant that is not at all vision transformer specific nor theoretically grounded.

Good suggestions! We agree that the introduction writing is over-detailed since most readers may have been aquainted with the background of 
vision transformers. We will trim this part and focus more on our research questions.

> The introduction is quite lengthy and involves several paragraph of generic discussions of vision transformers  ...

> Section 3: while it's interesting that the linear patch embedding allows us to describe the null space in terms of strict invariance, this analysis is no longer valid when we consider the entire model ...

Respectfully, we maintain that the study of linear patch embedding is important. As a perturbation that does not change the output of the patch embedding layer, will not change the prediction of the entire model. This nullspace is a subset of the set of perturbations that vision transformers are invariant to, 
and we study the non-linear nullspace to explore more.

> ... How does this method compare against training on random epsilon noise?

Good points! We did experiments to compare our method with random noise with different $epsilon$ values. The result is shown
in the below table. 

| $\epsilon$ |  method   | clean | FGSM  | DamageNet |   A   | C (⭣) |  V2   |   R   | Sketch | Stylized |
|:----------:|:---------:|:-----:|:-----:|:---------:|:-----:|:-----:|:-----:|:-----:|:------:|:--------:|
|    0.01    | nullspace | 77.44 | 26.04 |   33.65   | 20.45 | 56.26 | 66.47 | 41.4  | 23.34  |  15.85   |
 |    0.03    | nullspace | 77.47 | 25.95 |   32.43   | 20.77 | 55.98 | 66.5  | 41.61 | 25.67  |  16.02   |
|    0.1     | nullspace | 77.06 | 25.38 |   33.09   | 20.16 | 56.41 | 66.47 | 40.42 | 22.66  |  15.78   |

The result shows that our method is insensitive to the choice of $\epsilon$, and it
consistently outperforms the random noise baseline for most evaluation settings with different $\epsilon$ values, and
often by a large margin. The only exceptions are the clean accuracies and ImageNet-C. The latter is
because a lot of corruptions in ImageNet-C are some kind of random noise.

>  I suggest the authors tone down the claim “unexplored concept of nullspace…” ...

We are sorry for the inappropriate tone. In this paper, we study the robustness of vision transformers
from nullspace, an intrinsic property of the model, and enhance different kinds of robustness through this property, 
which we believe is a novel perspective. 
However, it is imprudent to say this concept is
"unexplored". We will replace it with "relatively under-studied".

> Equation 6 missing the optimization goals and constraints, which come later in lengthy discussion ...

We appreciate your careful review of our work. We will correct these details in the writing.


>  “fewer engineering hurdles” can you clarify what you mean by this?

We highlight ViTs are the foundation versions of the much complex architectures developed later on and in the corresponding paragraph, we attempt to justify our reasoning for only working with ViTs. The complexities introdcued by later models are usually in form of hybridisation with CNNs or alterations to the multi-head attention blocks. Given the immediate accessibility of the source codes, pre-trained weights and the wealth of available resources, there exists a compelling inclination to employ the Vision Transformer (ViT) models as opposed to any other much recent transformer based architecture. 

> Can you clarify what the statement of Proposition 1 Condition 2? Are you stating the column space of V is contained in the row space of QK^T for each attention head? And can you justify why this assumption holds?

Yes, we have to admit that Proposition 1 Condition 2 is strong. We established a sufficient condition
for the non-linear nullspace to exist, which does not necessarily hold for commonly used transformer models. As we
mentioned in line 173, 
it is very difficult to theoretically construct the non-linear nullspace or show its existence for
generic transformers. This is why we used an optimization method
to find elements in the nullspace. In addition, given the connections between nullspace and
robustness, enforcing this condition to enhance the robustness is a possible direction for future work.

> What’s the justification for Proposition 1. Condition 3?

The same applies to Proposition 1 Condition 3. This condition can be translated like this: given a set of matrices, there exists some row
in some matrix which lies in the row space of another matrix. This is not uncommon
in practice due to the limited numeric precision. For example, in our
numeric experiments in Table 1 in Appendix C, we find non-trivial nullspace of the patch embedding layer in ViT-B patch 16, though it has the
same input and output dimensions. This implies that some rows happen to be collinear in the weight matrix.

> Does the optimization in equation 6 only search for a single element statisfying closeness under addition? If so how many vectors do you learn and what are their values?

Yes, our optimization method searches for one nullspace vector at a time, which is at the boundary of the $\epsilon$-approximate nullspace
by our design. It is a vector with dimension given by (number of patches) $\times$ (number of hidden states). In each experiment
run, we search for 20 different vectors. 

> Figure 2 references epsilon on the x-axis, but epsilon is not defined in the optimization in equation 6.

We apologise for the confusion. $\epsilon$ is introduced later on in equation (7), consequently, we have shifted this ablation discussion to section 5.

> “as the epsilon criteria becomes smaller, the learnt noise becomes better and better” Is this conclusion based on the % of matching predictions? ...

This is based on both the metrics reported in the plot (1) matching predictions and (2) Mean-squared error of the confidence values of ground-truth categories.

> Our preliminary experiments indicate that there may exist a non-isomorphic space in the input space ...

What we meant is "non-isotropic", sorry for the typo. It means that the model is more sensitive
to perturbations in certain directions than others. In Figure 2 we used dashed line to show the model's performance
after permuting the elements of the noise vector (Line 202). The permutation preserves
the norm of the noise but randomly resets its direction, which results in strong influence to the model's predictions. This indicates that by gradient descent,
we learn noise vectors that are large enough to break the model if towards arbitrary direction,
but the direction is learned to produce little influence to the model.

> Equations 7 and 8 rehash the optimization you already present in Section 4.

About Equations 7 and 8, sorry for the repetition. We will re-organize the presentation of Figure 2 to avoid the repetition.

> I’m having a hard time following this logic. If the procedure you describe identifies vectors in the approximate null space, then by definition ...

Regarding Line 246, our intuition is that a robust model should be insensitive to some large perturbations as long as they do
not change the semantic information of the image. There should exist some directions in the input space (or the latent space 
after patch embedding) along which perturbations do not change the semantic information of the image for an object
classification task. We hypothesize that a robust model should be invariant to such perturbations. Our preliminary experiments only 
find nullspace vectors with some $\epsilon$ errors, which may indicate imperfect robustness, and by enforcing invariance to
these perturbations, we may enhance the robustness of the model. This is corroborated by our experiments.

> I’m quite surprised ViT-S accuracy improves so dramatically for ImageNet A and Sketch ...

The improvement of our method on ImageNet-A and ImageNet-sketch may be understood from the working of our method. 
We search for large nullspace noise that do not hurt the predictions of a pretrained
vision transformer model, which intuitively suggests that the noise selectively perturbs non-semantic information of the 
image. By iteratively find such perturbations and enforcing invariance to them, the model may be trained to unlearn the color 
and background bias. This may benefit generalization on these two datasets, as the important role
of background bias has been highlighted in the ImageNet-A paper [1], and ImageNet-Sketch radically removes the color information.

> This paper shows training with https://arxiv.org/abs/1808.08750 augmenting salt-and-pepper ...

We think that our result does not conflict with that of the previous work, 
because the experiment settings are fundamentally different: (1) In [2], the added salt-and-pepper noise is
so intense that it reduces the performance of a model trained on natural images to random. In contrast, we train the nullspace
noise to be sufficiently benign. (2) [2] trained the model only on the perturbed images, and this may be why 
the model has good performance on i.i.d test images (images with the same salt-and-pepper noise) but random performance on other distributions.
In contract, we add the loss on both the raw image and the perturbed image (Appendix B Algorithm 1 Line 19), as a common practice in
adversarial training, which enables the model to learn on both distributions. 

References:\
[1] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. CVPR 2021. \
[2] Robert Geirhos, Carlos R Medina Temme, Jonas Rauber, Heiko H Sch ̈ utt, Matthias Bethge, and Felix A Wichmann. Generalisation in humans 
and deep neural networks. NeurIPS 2018.

> The application to model patenting is not clear to me at all ...

The model patenting references to the exact nullspace noise which is derived from the linear patch embedding layer of the ViTs. It is true that nullspace noise might not be unique to a particular instance of the model, however, by generating a large number of such noise samples we can confidently pursue any violations. Cases where exact nullspace is not a viable solution, we can employ approximated noise in a similar fashion of synthesizing many samples and performing verificaiton.

> I’m not sure what insight this work sheds in terms of how transformer models are robust to input perturbations ...

The main conclusion of this paper is that the robustness of vision transformers is
related to the model's invariance to perturbations along certain directions. The
nullspace of the patch embedding layer provides such invariance, and the non-linear encoder 
also exhibits invariance with similar property to the nullspace in linear algebra.
We show that by enforcing such invariance through nullspace augmentation, we can
enhance the robustness of vision transformers on various test distributions.

> In the proof of proposition 1, what what r_{m, k}?

 As shown in Proposition 1 (Line 166), r_{m,k} denotes the $k$th row of the $m$th
matrix, which lies in the row space of another matrix as an instantiation of Condition 3.

> For the ViT-L 16x16, the computed nullspace dimension is 0 ...

Sorry for the confusion. As discussed in Line 137-140, when the
patch embedding layer has larger input dimension than output dimensions, the existence of
non-trivial nullspace is guaranteed by the rank-nullity theorem. The premise holds for
most of the common ViT configurations, but not all of them.

