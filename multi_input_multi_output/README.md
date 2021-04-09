#### Multi-Input Multi-Output (Multinet)

---

In this section we split up the network architectures to have a separate neural 
network per modality. Seeing that we are training with only two modalities (RGB and Depth) 
then we have only two networks. We can now take advantage of some unimodal self-supervised pretraining 
for each network to learn proper feature representations prior to training them concurrently on a classification 
task. The following are some self-supervised strategies we aim to implement and try out:

- Self-Supervised training as presented in Paper 1
- Unsupervised simCLR on each separate network
- Unsupervised Learning of Visual Features by Contrasting Cluster Assignments

We will also be experimenting with different base encoder neural network structures and try out all of the aforementioned 
self-supervised pretraining strategies on them. The following is a list of network options we are considering:

- ResNet50


Any hyperparameter tuning we perform will be based off the [Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner).

There are two possible options to train Multinet:

1 - Self-supervised pretraining for each base encoder separately, combine them to train a classifiaction
head while the base encoders are frozen and finally un-freezing the whole model for slow learning rate
fine tuning.

2 - No pretraining whatsoever, instead create Multinet and train both branches concurrently on the
            learning feature embeddings and classification.

##### Option 1 Results
| RGB Encoder  | RGB SSL           | Depth Encoder|    Depth SSL      | Learning Rate       |Validation Accuracy |
| -------------|:-----------------:| ------------:|------------------:|--------------------:|-------------------:|
| ResNet50     |Unsupervised simCLR| ResNet50     |Unsupervised simCLR|0.001 (Cosine Decay) |         X%         |

---