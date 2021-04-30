

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2

model = VGG16(weights='imagenet')

def classify_vgg(img):
    x = cv2.resize(img, (224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    dec_preds = decode_predictions(preds, top=3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    linepos = [12, 24, 36]
    # print top 3 results
    for i in range(3):
        cv2.putText(img, '{}: {:.3f}'.format(dec_preds[0][i][1],
                                             dec_preds[0][i][2]),
                    (10, linepos[i]),
                    font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return img