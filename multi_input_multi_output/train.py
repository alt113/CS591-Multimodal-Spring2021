# import the necessary packages
import os

from multi_input_multi_output.models import multi_input_multi_output_model
from tensorflow.keras.optimizers import Adam

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)
PATH_TO_CWD = os.getcwd()

model = multi_input_multi_output_model()

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
    "rgb_output": "categorical_crossentropy",
    "depth_output": "categorical_crossentropy"
}
loss_weights = {"rgb_output": 1.0, "depth_output": 1.0}

# initialize the optimizer and compile the model
# print("[INFO] compiling model...")
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])
#
# # train the network to perform multi-output classification
# H = model.fit(x=[trainRGBX, trainDepthX],
#               y={
#                   "category_output": trainCategoryY, "color_output": trainColorY
#               },
#               validation_data=(testX, {"category_output": testCategoryY, "color_output": testColorY}),
#               epochs=EPOCHS,
#               verbose=1)
# # save the model to disk
# print("[INFO] serializing network...")
# model.save(PATH_TO_CWD, save_format="h5")

