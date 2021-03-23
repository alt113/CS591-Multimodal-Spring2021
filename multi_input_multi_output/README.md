#### Multi-Input Multi-Output

---

In this section we split up the network architectures to have a separate neural 
network per modality. Seeing that we are training with only two modalities (RGB and Depth) 
then we have only two networks. We can now take advantage of some unimodal self-supervised pretraining 
for each network to learn proper feature representations prior to training them concurrently on a classification 
task. The following are some self-supervised strategies we aim to implement and try out:

- Self-Supervised training as presented in Paper 1
- Supervised SimCLR on each separate network
- Unsupervised SimCLR on each separate network
- Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
- Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning

We will also be experimenting with different base encoder neural network structures and try out all of the aforementioned 
self-supervised pretraining strategies on them. The following is a list of network options we are considering:

- LeNet-5
- AlexNet
- VGG (16 and 19)
- Inception
- ResNet(50<sup>1</sup>, 101, and 152)


Any hyperparameter tuning we perform will be based off the [Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner).

The following is a summary of the results we obtained after performing self-supervised pretraining and then fine-tuning 
the models on the image classification task.

```
INSERT TABLE HERE
```

---
<sup>1</sup>ResNet50 is the baseline network architecture.