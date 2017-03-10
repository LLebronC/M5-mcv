# Keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D

from keras.applications.inception_v3 import InceptionV3
from models.inception_v3Reg import InceptionV3Reg

# Paper: https://arxiv.org/pdf/1409.1556.pdf

def build_inceptionV3(img_shape=(3, 224, 224), n_classes=1000, l2_reg=0., load_pretrained=False,
                                   freeze_layers_from='base_model'):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model
    if l2_reg == 0.:
        base_model = InceptionV3(include_top=False, weights=weights,
                           input_tensor=None, input_shape=img_shape)
    else:
        print "Inceptionv3 Reg"
        base_model = InceptionV3Reg(include_top=False, weights=weights,
                           input_tensor=None, input_shape=img_shape, l2_reg=l2_reg)
    # Add final layers
    x = base_model.output
    x = AveragePooling2D((4, 4), strides=(4, 4), name='avg_pool')(x)
    x = Flatten(name="flatten")(x)
    x = Dense(n_classes, name='dense_1')(x)
    predictions = Activation("softmax", name="softmax")(x)

    # This is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True

    return model