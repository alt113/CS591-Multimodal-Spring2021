#### Shared Weights

---

Paper 1 (Improving Unimodal Object Recognition with Multimodal Contrastive Learning) 
aims to perform its multimodal training using a Siamese network architecture. 
The loss the paper uses is the contrastive loss. In this directory, we implement the 
same architecture and loss as in Paper 1 and try out several variants (e.g. Binary Cross Entropy). The following 
are the results that we achieve along with the configuration settings for our Siamese network:

| Base Encoder  | Loss          |Validation Accuracy|
| ------------- |:-------------:| -----:|
| ResNet50      | BCE           | X %   |
| ResNet50      | Contrastive Loss      |   X% |
| ?             | BCE           |    X% |
| ?             | Contrastive Loss      |    X% |

In each of the above configurations we used the following training hyperparameters:

- Adam Optimizer
- Batch Size of 64
- Trained for 100 epochs