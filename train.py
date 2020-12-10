import os
import keras
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import sys
import pickle
from keras.utils import plot_model
from keras.applications.xception import Xception
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.xception import preprocess_input
from keras.models import Model
import string
import numpy as np
from pickle import load
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM ,GRU
from keras.layers import Embedding
from keras.layers import Dropout, Reshape, Lambda, Concatenate
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from nltk.translate.bleu_score import corpus_bleu
from keras.callbacks import ReduceLROnPlateau

emb_dim = 50
batch_size = 32
snaphot_folder = '../weights'

def progressBar(value, endvalue, bar_length=20,job='Job'):

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0} Completion: [{1}] {2}%".format(job,arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return list(set(dataset))

def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

#Reading the captions
def get_captions():
  doc = load_doc("Data/Flickr8k_text/Flickr8k.token.txt")
  descriptions = {}
  for line in doc.split('\n'):
    try:
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        if image_id not in descriptions:
            descriptions[image_id] = list()
        descriptions[image_id].append(image_desc)
    except :
        print(line)
  return descriptions     

def clean_data(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>0]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] =  ' '.join(desc)
            
    return descriptions

start_token = '<startseq>'
end_token = '<endseq>'
def add_end_start_tokens(descriptions):
    for key in descriptions:
        for i in range(len(descriptions[key])):
            descriptions[key][i] = start_token + ' ' + descriptions[key][i] + ' ' + end_token
    return descriptions

def load_train_test(descriptions, dataset):
    dataset_ = {}
    for image_id in dataset:
        dataset_[image_id] = descriptions[image_id]
        
    return dataset_

def generate_photo_feature():
  descriptions = get_captions()
  clean_descriptions = clean_data(descriptions)
  descriptions_tokenSE = add_end_start_tokens(clean_descriptions)

  train_imgs_addr = 'Data/Flickr8k_text/Flickr_8k.trainImages.txt'
  test_imgs_addr = 'Data/Flickr8k_text/Flickr_8k.testImages.txt'
  dev_imgs_addr = 'Data/Flickr8k_text/Flickr_8k.devImages.txt'

  train_imgs_names = load_set(train_imgs_addr)
  test_imgs_names = load_set(test_imgs_addr)
  dev_imgs_names = load_set(dev_imgs_addr)

  len(train_imgs_names), len(test_imgs_names), len(dev_imgs_names)

  train_descriptions = load_train_test(descriptions_tokenSE, train_imgs_names)
  dev_descriptions = load_train_test(descriptions_tokenSE, dev_imgs_names)
  test_descriptions = load_train_test(descriptions_tokenSE, test_imgs_names)
  return train_descriptions, dev_descriptions, test_descriptions

def get_image_feature_model():
  xception = Xception()
  extractor = Model(inputs=xception.inputs, outputs=xception.layers[-2].output) # removing 2 last fully connected layers
  return extractor

def extract_features(description, model, inpute_size = (229,229)):
    directory = 'Flicker8k_Dataset'
    features = {}
    i = 0
    for name in description:
        progressBar(value=i,endvalue=len(description))
        i +=1
        filename =os.path.join(directory, name+'.jpg')
        image = load_img(filename, target_size=inpute_size)
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature.reshape(2048)
    return features

def get_train_test_dev_features():
  #extractor = get_image_feature_model()
  #train_descriptions, dev_descriptions, test_descriptions = generate_photo_feature()
  # train_features = extract_features(train_descriptions, extractor)
  # test_features = extract_features(test_descriptions, extractor)
  # dev_features = extract_features(dev_descriptions, extractor)

  # pickle.dump(train_features, open('train_features.pkl', 'wb'))
  with open('train_features.pkl','rb') as f:
    train_features= pickle.load(f)

  # pickle.dump(test_features, open('test_features.pkl', 'wb'))
  with open('test_features.pkl','rb') as f:
    test_features= pickle.load(f)

  # pickle.dump(dev_features, open('dev_features.pkl', 'wb'))
  with open('dev_features.pkl','rb') as f:
    dev_features= pickle.load(f)
  return train_features, dev_features, test_features

#get vocab and vocab_size
# making a vocabulary of the words that occur more than word_count_threshold time
def create_reoccurring_vocab(descriptions, word_count_threshold = 10):
    # Create a list of all the captions
    all_captions = []
    for key, val in descriptions.items():
        for cap in val:
            all_captions.append(cap)

    # Consider only words which occur at least 10 times in the corpus
    word_counts = {}
    nsents = 0
    for sent in all_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    return vocab

def get_vocab_size_and_indexing(train_descriptions):
    vocab = create_reoccurring_vocab(train_descriptions, word_count_threshold = 5)
    oov_token = '<UNK>'
    filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n' # making sure all the last non digit non alphabet chars are removed
    tokenizer = keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)
    tokenizer.fit_on_texts(vocab)
    vocab_size = len(tokenizer.word_index) + 1
    print('vocab_size :', vocab_size)

    ixtoword = {} # index to word dic
    wordtoix = {} # word to index dic

    tokenizer.word_index['<PAD0>'] = 0 # no word in vocab has index 0. but padding is indicated with 0
    wordtoix = tokenizer.word_index # word to index dic

    for w in tokenizer.word_index:
      ixtoword[tokenizer.word_index[w]] = w

    return vocab_size, ixtoword, wordtoix, vocab

# finding the maximum length of questions and answers
# because there are senteces with unusually long lengths,
# we caculate the max length that p% of data can be placed in
def get_max_length(desc,p):
    all_desc = []
    # Create a list of all the captions
    for i in desc:
        for j in desc[i]:
            all_desc.append(j)

    length_all_desc = list(len(d.split()) for d in all_desc)

    print('percentile {} of len of questions: {}'.format(p,np.percentile(length_all_desc, p)))
    print('longest sentence: ', max(length_all_desc))

    return int(np.percentile(length_all_desc, p))

# outputs a batch of captions-pictures
def data_generator(descriptions, photos):

    #des = get_captions()
    vocab = create_reoccurring_vocab(descriptions, word_count_threshold = 5)
    oov_token = '<UNK>'
    filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n' # making sure all the last non digit non alphabet chars are removed
    tokenizer = keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)
    tokenizer.fit_on_texts(vocab)

    max_length = get_max_length(descriptions, 90)
    X1, X2, y = [], [], []
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key]
            for desc in desc_list:
                # find the index of each word of the caption in vocabulary
                seq = tokenizer.texts_to_sequences(desc.split())
                # split one sequence into multiple X, y pairs
                # Each step of the following for loop selects one word
                # from the caption, consider that word as y and
                # all the words before that will be the X
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i] # words until i are inseq word i is outseq
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n == batch_size:
                yield [np.array(X1), np.array(X2).squeeze(axis=-1)], np.array(y).squeeze(axis=-1)
                X1, X2, y = list(), list(), list()
                n=0

def make_embedding_layer(train_descriptions, embedding_dim=50, glove=True):
    if glove == False:
        print('Just a zero matrix loaded')
        embedding_matrix = np.zeros((vocab_size, embedding_dim)) # just a zero matrix
    else:
        glove_dir = './glove.6B/'
        embeddings_index = {}
        # f = open(os.path.join(glove_dir, 'glove.6B.'+str(embedding_dim)+'d.txt'), encoding="utf-8")
        # for line in f:
        #     values = line.split()
        #     word = values[0]
        #     coefs = np.asarray(values[1:], dtype='float32')
        #     embeddings_index[word] = coefs
        # f.close()
        # Get x-dim dense vector for each of the vocab_rocc

        vocab_size, ixtoword, wordtoix, vocab = get_vocab_size_and_indexing(train_descriptions)
        # max_length = max_length(desc, 90)

        embedding_matrix = np.zeros((vocab_size, embedding_dim)) # to import as weights for Keras Embedding layer
        for word, i in wordtoix.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in the embedding index will be all zeros
                embedding_matrix[i] = embedding_vector
        print('GloVe loaded!')

    embedding_layer = Embedding(vocab_size, embedding_dim, mask_zero=True, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])

    return embedding_layer

def make_embedding_layer_without_glove(train_descriptions, embedding_dim=50, glove=True):

    embeddings_index = {}
    vocab_size, ixtoword, wordtoix, vocab = get_vocab_size_and_indexing(train_descriptions)
    embedding_matrix = np.zeros((vocab_size, embedding_dim)) # to import as weights for Keras Embedding layer

    for word, i in wordtoix.items():
      embedding_vector = i
      if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(vocab_size, embedding_dim, mask_zero=True, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])

    return embedding_layer

# for masking pad0 outputs
def masked_loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = K.sparse_categorical_crossentropy(real, pred, from_logits= False) # sparse cat gets pred classes in 'int' form
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

from keras.layers import RepeatVector

def make_model(train_descriptions, max_length, vocab_size, dout= 0.2, feature_size= 2048, units= 256):
    embedding = make_embedding_layer_without_glove(train_descriptions, emb_dim, glove=True)
    features = Input(shape=(feature_size,)) # output size of feature extractor
    X_fe_one_dim = Dense(units, activation='relu')(features) # because i have used bidirectional LSTM, the number of units should
                                                   # become double here in order for the add function to work
    X_fe = RepeatVector(max_length)(X_fe_one_dim)
    X_fe = Dropout(dout)(X_fe)

    seq = Input(shape=(max_length,))
    X_seq = embedding(seq)
    X_seq = Lambda(lambda x: x, output_shape=lambda s:s)(X_seq) # remove mask from the embedding cause concat doesn't support it
    X_seq = Dropout(dout)(X_seq)
    X_seq = Concatenate(name='concat_features_word_embeddings', axis=-1)([X_fe,X_seq])
    X_seq = GRU(units, return_sequences=True)(X_seq,initial_state=X_fe_one_dim) # passing features as init_state
    X_seq = Dropout(dout + 0.2)(X_seq)
    X_seq = GRU(units, return_sequences=False)(X_seq)

    outputs = Dense(vocab_size, activation='softmax')(X_seq)

    # merge the two input models
    model = Model(inputs=[features, seq], outputs = outputs, name='model_with_features_each_step')
    return model

# generate a description for an image greedy way
def generate_desc(max_length, model, photo_fe, inference= False):

    des = get_captions()
    descriptions = clean_data(des)

    #fetching tokenizer
    vocab = create_reoccurring_vocab(descriptions, word_count_threshold = 5)
    oov_token = '<UNK>'
    filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n' # making sure all the last non digit non alphabet chars are removed
    tokenizer = keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)
    tokenizer.fit_on_texts(vocab)

    ixtoword = {} # index to word dic
    wordtoix = {} # word to index dic

    tokenizer.word_index['<PAD0>'] = 0 # no word in vocab has index 0. but padding is indicated with 0
    wordtoix = tokenizer.word_index # word to index dic

    for w in tokenizer.word_index:
      ixtoword[tokenizer.word_index[w]] = w

    # seed the generation process
    in_text = start_token
    # iterate over the whole length of the sequence
    # generate one word at each iteratoin of the loop
    # appends the new word to a list and makes the whole sentence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences(in_text.split()) #[wordtoix[w] for w in in_text.split() if w in wordtoix]
        # pad input
        photo_fe = photo_fe.reshape((1,2048))
        sequence = pad_sequences([sequence], maxlen=max_length).reshape((1,max_length))
        # predict next word
        yhat = model.predict([photo_fe,sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = ixtoword[yhat]
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next v
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == end_token:
            break

    if inference == True:
        in_text = in_text.split()
        if len(in_text) == max_length:
            in_text = in_text[1:] # if it is already at max len and endseq hasn't appeared
        else:
            in_text = in_text[1:-1]
        in_text = ' '.join(in_text)

    return in_text

# calculating BLEU score of predictions
def evaluate_model(max_length, model, descriptions, photos_fe, K_beams= 1, log=False):
    actual, predicted = list(), list()
    # step over the whole set
    i=0
    for key, desc_list in descriptions.items():
        # generate description
        i+=1
        progressBar(i, len(descriptions), bar_length=20,job='Evaluating')
        if K_beams == 1:
            yhat = generate_desc(max_length, model, photos_fe[key])
        else:
            yhat=beam_search_pred(model, photos_fe[key], wordtoix, K_beams = K_beams,log=log)

        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())

    # calculate BLEU score
    b1=corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    b2=corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    b3=corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    b4=corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    print('\n')
    print('BLEU-1: %f' % b1)
    print('BLEU-2: %f' % b2)
    print('BLEU-3: %f' % b3)
    print('BLEU-4: %f' % b4)
    return [b1,b2,b3,b4]
# def train(model, train_features, dev_features, steps, vocab_size):

#   Reduce_lr=ReduceLROnPlateau(monitor='loss', factor=0.9, patience=5, verbose=0, mode='auto', min_delta=0.0001, min_lr=0.000001)
#   model.compile(loss=masked_loss_function, optimizer= 'adam')
#   history={'loss':[], 'BLEU_val':[]}

#   for i in range(1,100):
#     print('Epoch :',i,'\n')
#     # create the data generator
#     generator = data_generator(train_descriptions, train_features)
#     # fit for one epoch
#     h = model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1, callbacks=[Reduce_lr] )
#     ep = i + 1
#     history['loss'].append(h.history['loss'])

#     # save model every 10 epochs
#     if i % 20 == 0:
#         #test()
#         #model.save_weights(snaphot_folder+'/model_' + str(i) + '.h5')
#         #belus= evaluate_model(model, dev_descriptions, dev_features, K_beams=1)
#         history['BLEU_val'].append((belus,i))

#     print('\n','='*80)

def main():
  train_descriptions, dev_descriptions, test_descriptions = generate_photo_feature()
  vocab_size, ixtoword, wordtoix, vocab = get_vocab_size_and_indexing(train_descriptions)
  max_length = get_max_length(train_descriptions, 90)
  model = make_model(train_descriptions, max_length, vocab_size)

  train_features, dev_features, test_features = get_train_test_dev_features()
  #steps = len(train_descriptions)//batch_size
  #train(model, train_features, dev_features, steps, vocab_size)

  Reduce_lr=ReduceLROnPlateau(monitor='loss', factor=0.9, patience=5, verbose=0, mode='auto', min_delta=0.0001, min_lr=0.000001)
  model.compile(loss=masked_loss_function, optimizer= 'adam')
  history={'loss':[], 'BLEU_val':[]}

  for i in range(1,5):
    print('Epoch :',i,'\n')
    # create the data generator
    generator = data_generator(train_descriptions, train_features)
    # fit for one epoch

    steps = len(train_descriptions)//batch_size

    h = model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1, callbacks=[Reduce_lr] )
    ep = i + 1
    #with open('output_75_32.txt', 'a') as f:
    #  f.write(str(h.history['loss'][0]) + '\n')
    history['loss'].append(h.history['loss'])

    #save model every 10 epochs
    if i % 4 == 0:
         #test()
        model.save_weights(snaphot_folder+'/model_' + str(i) + '.h5')
        #belus= evaluate_model(max_length, model, dev_descriptions, dev_features, K_beams=1)
        #history['BLEU_val'].append((belus,i))

    print('\n','='*80)
  print(h.history['loss'])

  evaluate_model(max_length, model, dev_descriptions, dev_features, K_beams=1)

if __name__ == "__main__":
    main()
