"""
    In order to build our multi-input network we will need two branches:
        - The first branch will be a CNN for the RGB modality
        - The Second branch will also be a CNN but for the depth modality

    These branches will then carry out their own respective classification on the
    input dataset and we will end up having multi-output classification with multi-loss functions
"""
# import the necessary packages
from tensorflow import keras
from keras.models import Model

import numpy as np

CLS_RGB = 0.
CLS_DEPTH = 0.
RGBS_TURN = True


def multi_input_multi_output_model(rgb_classifier, rgb_branch, depth_classifier, depth_branch):
    """
        Two base encoder with separate classifier tops. This does not perform any
        self-supervised pretraining but instead trains both encoders together from scratch.
    """
    return Model(inputs=[rgb_classifier.input, depth_classifier.input],
                 outputs=[rgb_branch, depth_branch], name='multinet')


class MultiNet(object):

    def __init__(self,
                 rgb_classifier,
                 rgb_output_branch,
                 depth_classifier,
                 depth_output_branch,
                 opt='adam'):
        # RGB model (entire)
        self.rgb = rgb_classifier

        # RGB output branch
        self.rgb_branch = rgb_output_branch

        # Depth model (entire)
        self.depth = depth_classifier

        # Depth output branch
        self.depth_branch = depth_output_branch

        # Model
        self.model = self.build_model()

        # Print model summary
        self.model.summary()

        # Optimizer
        if opt == 'adam':
            self.optimizer = keras.optimizers.Adam(lr=0.001)
        elif opt == 'sgd':
            self.optimizer = keras.optimizers.SGD(lr=0.001)
        else:
            raise ValueError('Unsupported Optimizer! Only Adam and SGD allowed.')

        # Loss Function
        self.loss_func = self.model_loss()

    def build_model(self):
        return multi_input_multi_output_model(rgb_classifier=self.rgb,
                                              rgb_branch=self.rgb_branch,
                                              depth_classifier=self.depth,
                                              depth_branch=self.depth_branch)

    def model_loss(self):

        rgb_weights = self.rgb.get_layer('resnet50v2_rgb').get_weights()[-10:]
        depth_weights = self.depth.get_layer('resnet50v2_depth').get_weights()[-10:]

        rgb_corr = []
        depth_corr = []
        for rw, dw in zip(rgb_weights, depth_weights):

            if len(rw.shape) == 1:
                rgb_corr.append(np.dot(rw, rw))
                depth_corr.append(np.dot(dw, dw))
            else:
                rw = np.squeeze(rw)
                dw = np.squeeze(dw)
                rgb_corr.append(np.dot(rw, rw.T))
                depth_corr.append(np.dot(dw, dw.T))

        rgb_corr = np.array(rgb_corr)
        depth_corr = np.array(depth_corr)

        frob_norm_squared_rgb = np.square(np.linalg.norm(rgb_corr - depth_corr, ord=None))
        frob_norm_squared_depth = np.square(np.linalg.norm(depth_corr - rgb_corr, ord=None))

        def multinet_loss(y_true, y_pred):

            global CLS_RGB
            global CLS_DEPTH
            global RGBS_TURN

            scce = keras.losses.SparseCategoricalCrossentropy()

            # y_true = y_true.numpy().T
            # y_pred = y_pred.numpy()
            if RGBS_TURN and CLS_RGB == 0.:
                RGBS_TURN = False
                CLS_RGB = scce(y_true, y_pred).numpy()
                return scce(y_true, y_pred)
            elif not RGBS_TURN and CLS_DEPTH == 0.:
                RGBS_TURN = True
                CLS_DEPTH = scce(y_true, y_pred).numpy()
                return scce(y_true, y_pred)

            elif RGBS_TURN:
                RGBS_TURN = False
                delta_l_rgb = CLS_RGB - CLS_DEPTH
                # check for RGB negative flow
                if delta_l_rgb > 0:
                    ro_rgb = np.exp(2 * delta_l_rgb) - 1
                else:
                    ro_rgb = 0
                    
                CLS_RGB = scce(y_true, y_pred).numpy()
                return scce(y_true, y_pred) + 0.01 * ro_rgb * frob_norm_squared_rgb
                
            else:
                RGBS_TURN = True
                delta_l_depth = CLS_DEPTH - CLS_RGB
                # check for Depth negative flow
                if delta_l_depth > 0:
                    ro_depth = np.exp(2 * delta_l_depth) - 1
                else:
                    ro_depth = 0

                CLS_DEPTH = scce(y_true, y_pred).numpy()
                return scce(y_true, y_pred) + 0.01 * ro_depth * frob_norm_squared_depth

        return multinet_loss

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_func, run_eagerly=True)

        print('Model Compiled!')
