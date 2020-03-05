import os, argparse
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
K.set_image_data_format('channels_first')
K.set_image_dim_ordering('th')
__import__('sklearn.externals.joblib.numpy_pickle')

VQA_weights_file_name   = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = 'models/CNN/vgg16_weights_th_dim_ordering_th_kernels.h5'

verbose = 1

def get_image_model(CNN_weights_file_name):
    from models.CNN.VGG import VGG_16
    image_model = VGG_16(CNN_weights_file_name)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return image_model

model_vgg = get_image_model(CNN_weights_file_name)

def get_image_features(image_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the
    weights (filters) as a 1, 4096 dimension vector '''
    image_features = np.zeros((1, 4096))
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    image_features[0,:] = model_vgg.predict(im)[0]
    return image_features

def get_question_features(question):
    word_embeddings = spacy.load('en_vectors_web_lg')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in xrange(len(tokens)):
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor

def get_VQA_model(VQA_weights_file_name):
    from models.VQA.VQA import VQA_MODEL
    vqa_model = VQA_MODEL()
    vqa_model.load_weights(VQA_weights_file_name)
    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_file_name', type=str, default='static/images/acc/gateway.jpg')
    parser.add_argument('-question', type=str, default='What is in the picture?')
    args = parser.parse_args()

    if verbose : print("\n\n\nLoading image features ...")
    image_features = get_image_features(args.image_file_name)

    if verbose : print("Loading question features ...")
    question_features = get_question_features(unicode(args.question, 'utf-8'))

    if verbose : print("Loading VQA Model ...")
    vqa_model = get_VQA_model(VQA_weights_file_name)

    if verbose : print("\n\n\nPredicting result ...")
    y_output = vqa_model.predict([question_features, image_features])

    y_sort_index = np.argsort(y_output)

    labelencoder = joblib.load(label_encoder_file_name)
    print("You: ",type(labelencoder))
    print("You: ",labelencoder.shape)

    opfilelabel = open('outputlabel.txt', 'w')
    opfilecon = open('outputcon.txt', 'w')

    for label in reversed(y_sort_index[0,-5:]):
        print("something: ", y_output[0,label])
        con=str(round(y_output[0,label]*100,2))+'%'
        lab=labelencoder.inverse_transform(label)
        print(con+"% - "+lab)
        opfilelabel.write(lab)
        opfilelabel.write("\n")
        opfilecon.write(con)
        opfilecon.write("\n")

    opfilelabel.close()
    opfilecon.close()


if __name__ == "__main__":
    main()
