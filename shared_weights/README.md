#### Shared Weights

---

Paper 1 (Improving Unimodal Object Recognition with Multimodal Contrastive Learning) 
aims to perform its multimodal training using a Siamese network architecture. 
The loss the paper uses is the contrastive loss. In this directory, we implement the 
same architecture and loss as in Paper 1 and try out another variant with the Binary Cross Entropy (BCE) loss instead. The following 
are the results that we achieve along with the configuration settings for our Siamese network:

| Base Encoder        | Loss            |Validation Accuracy|
| --------------------|:---------------:| -----------------:|
| ResNet50            | BCE             | X%                |
| ResNet50<sup>*</sup>| Contrastive Loss| X%                |
| ResNet101           | BCE             | X%                |
| ResNet101           | Contrastive Loss| X%                |
| VGG16               | BCE             | X%                |
| VGG16               | Contrastive Loss| X%                |
| VGG19               | BCE             | X%                |
| VGG19               | Contrastive Loss| X%                |


In each of the above configurations we used the following encoder pre-training hyperparameters:

- Adam Optimizer
- Batch Size of 64
- Fixed learning rates of 0.001
- Run for 50 epochs

---
<sup>*</sup> Paper 1 benchmark