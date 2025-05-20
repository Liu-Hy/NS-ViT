# Approximate Nullspace Augmented Finetuning for Robust Vision Transformers
Official Codebase for CPAL 2025 (Oral) paper.
>[__"Approximate Nullspace Augmented Finetuning for Robust Vision Transformers"__](https://arxiv.org/abs/2403.10476)<br>
>Haoyang Liu, Aditya Singh, Yijiang Li, Haohan Wang<br>
[`[Paper]`](https://arxiv.org/abs/2403.10476) [`[Code]`](https://github.com/Liu-Hy/NS-ViT) [`[Project Page]`]()

***Abstract.***
> Enhancing the robustness of deep learning models, particularly in the realm of vision transformers (ViTs), is crucial for their real-world deployment. In this work, we provide a finetuning approach to enhance the robustness of vision transformers inspired by the concept of nullspace from linear algebra. Our investigation centers on whether a vision transformer can exhibit resilience to input variations akin to the nullspace property in linear mappings, which would imply that perturbations sampled from this nullspace do not influence the model's output when added to the input. We start from the observation that many existing ViTs satisfy this property because their patch embedding layer has a non-trivial nullspace. Then, we extend the notion of nullspace to nonlinear settings and demonstrate that it is possible to synthesize approximate nullspace elements for ViT's encoder blocks through optimization. Finally, we propose a finetuning strategy for ViTs wherein we augment the training data with synthesized approximate nullspace noise. We find that our finetuning approach significantly improves the models' robustness to both adversarial and natural image perturbations.

## Results

## Getting Started

### Installation

### Usage

## Citation
If you find this project useful, please consider citing:
```bibtex
@article{liu2024approximate,
  title={Approximate Nullspace Augmented Finetuning for Robust Vision Transformers},
  author={Liu, Haoyang and Singh, Aditya and Li, Yijiang and Wang, Haohan},
  journal={arXiv preprint arXiv:2403.10476},
  year={2024}
}
```
