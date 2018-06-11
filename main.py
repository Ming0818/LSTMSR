print("INITIALIZING PROGRAM...")


import os                   # for directories
import datetime             # for timestamping
import pickle               # for storing objects
import subprocess           # for sending shell commands
import numpy                # for matrix calculations
import sys                  # for stdout flush
import random               # for shuffling array / list contents
from scipy.special import expit     # for sigmoid squash
from PIL import Image       # for converting gray scale image into numbers


class Model:                # for storing LSTM models
    def __init__(self, m_learning_rate, m_hidden_unit_size):
        self.learning_rate = m_learning_rate            # adjustable parameter
        self.hidden_unit_size = m_hidden_unit_size      # adjustable parameter
        self.accuracy = 0                               # tested parameter
        temp_name = "{:.10f}".format(self.learning_rate).rstrip("0")       # float formatting for LR
        self.name = "lr{}h{}".format(temp_name[2:], self.hidden_unit_size)  # for identifying models
        # 4 gates, each a dxn matrix to handle input, and each with a dxd matrix for recursion
        #       at     wc uc                d = hidden_unit_size    n = input_size
        # zt = [it] = [wi ui] x [ xt ]      a, i, f, o = gate value
        #       ft     wf uf     ht-1       w, u = weight matrices
        #       ot     wo uo                x = input               h = recursion value
        # initialized value randomly with range [-0.5, 0.5)
        self.wm_hidden = numpy.random.random((self.hidden_unit_size*4, input_size+self.hidden_unit_size)) - 0.5
        # final part of recursion is pushed into a oxd matrix to generate output results
        # (total_speakers x hidden_unit_size)
        self.wm_score = numpy.random.random((Speaker.total_speakers, self.hidden_unit_size)) - 0.5

    def get_name(self):
        return self.name

    def get_learning_rate(self):
        return self.learning_rate

    def get_hidden_unit_size(self):
        return self.hidden_unit_size

    def set_accuracy(self, m_accuracy):
        self.accuracy = m_accuracy

    def get_accuracy(self):
        return self.accuracy

    def set_wm_hidden(self, m_wm_hidden):
        self.wm_hidden = m_wm_hidden

    def get_wm_hidden(self):
        return self.wm_hidden

    def set_wm_score(self, m_wm_score):
        self.wm_score = m_wm_score

    def get_wm_score(self):
        return self.wm_score


class Speaker:              # for storing speaker data
    total_speakers = 0       # keeping track of number of registered speakers

    def __init__(self, s_dialect, s_sex, s_id, s_audios):
        self.dialect = s_dialect        # DR1 - DR8
        self.sex = s_sex                # M || F
        self.speaker_id = s_id          # 3 characters 1 number
        self.audios = s_audios          # array of audios
        self.index = None              # speaker's target index
        Speaker.total_speakers += 1     # add total speakers after initialize
        # refreshes audio usage distribution per speaker initialized
        Audio.total_validate_SI = 0
        Audio.total_validate_SX = 0
        Audio.total_test_SI = 0
        Audio.total_test_SX = 0

    def get_dialect(self):
        return self.dialect

    def get_sex(self):
        return self.sex

    def get_id(self):
        return self.speaker_id

    def get_audios(self):
        return self.audios

    def get_name(self):
        return self.sex + self.speaker_id

    def set_index(self, s_index):
        self.index = s_index

    def get_index(self):
        return self.index


class Audio:                # for storing each audio file
    total_audios = 0        # keep track of number of audios registered
    total_validate_SI = 0   # for distributing audios into datasets
    total_validate_SX = 0
    total_test_SI = 0
    total_test_SX = 0

    def __init__(self, a_type, a_number, a_path_raw, a_path_converted):
        self.sentence_type = a_type                 # SA || SI || SX
        self.sentence_number = a_number             # 4 digit number
        self.path_raw = a_path_raw                  # path of source audio file
        self.path_converted = a_path_converted      # path of target image file
        self.index = None                          # speaker's target index
        self.data = None                           # image data in numerical value [0, 1]
        Audio.total_audios += 1                     # add total audio after initialize
        if self.sentence_type == "SX" and Audio.total_validate_SX < 1:      # 1 SX audio for validate set
            self.usage = "Validate"
            Audio.total_validate_SX += 1
        if self.sentence_type == "SX" and Audio.total_test_SX < 1:          # 1 SX audio for test set
            self.usage = "Test"
            Audio.total_test_SX += 1
        elif self.sentence_type == "SI" and Audio.total_validate_SI < 1:    # 1 SI audio for validate set
            self.usage = "Validate"
            Audio.total_validate_SI += 1
        elif self.sentence_type == "SI" and Audio.total_test_SI < 1:        # 1 SI audio for test set
            self.usage = "Test"
            Audio.total_test_SI += 1
        else:
            self.usage = "Train"    # 6 audio (2 SA, 3 SI, 1 SX) train, 2 (1 SI, 1 SX) validate, 2 (1 SI, 1 SX) test

    def get_type(self):
        return self.sentence_type

    def get_number(self):
        return self.sentence_number

    def get_path_raw(self):
        return self.path_raw

    def get_path_converted(self):
        return self.path_converted

    def get_usage(self):
        return self.usage

    def set_index(self, a_index):
        self.index = a_index

    def get_index(self):
        return self.index

    def set_data(self, a_data):
        self.data = a_data

    def get_data(self):
        return self.data


def train(f_dataset_train, h_prev, m_prev, t_index):
    inp_unit = f_dataset_train[:, 0].reshape(input_size, 1)                     # input vector xt taken from image data
    inp_whole = numpy.vstack((inp_unit, h_prev))                            # input vector It combined from xt and ht-1
    hidden_whole = numpy.dot(curr_model.get_wm_hidden(), inp_whole)             # raw hidden value for all gates zt
    a_gate = numpy.tanh(hidden_whole[:curr_model.get_hidden_unit_size(), :])    # split, squashed value from zt
    i_gate = expit(hidden_whole[curr_model.get_hidden_unit_size():2*curr_model.get_hidden_unit_size(), :])
    f_gate = expit(hidden_whole[2*curr_model.get_hidden_unit_size():3*curr_model.get_hidden_unit_size(), :] + f_bias)
    o_gate = expit(hidden_whole[3*curr_model.get_hidden_unit_size():, :])
    m_curr = i_gate * a_gate + f_gate * m_prev      # memory cell value ct calculated from at, it, ft, and ct-1
    h_curr = o_gate * numpy.tanh(m_curr)            # current timestep output value calculated from ot and ct
    if f_dataset_train[:, 1:].size == 0:            # on the last step, calculate score, errhc
        score = numpy.dot(curr_model.get_wm_score(), h_curr)            # output layer producing 24 length vector s
        score_prob = numpy.exp(score) / numpy.sum(numpy.exp(score))     # softmax layer producing probability vector P
        loss = -numpy.log(score_prob[t_index])                          # negative log loss error value E
        jac = numpy.diagflat(score_prob) - numpy.dot(score_prob, score_prob.T)  # matrix J for deriving softmax layer
        err_score = numpy.dot(jac, score_prob - target_array[t_index])  # error vector ds from J and audio target
        err_wm_score = numpy.dot(err_score, h_curr.T)                   # matrix error dws calculated from ds
        curr_model.set_wm_score(curr_model.get_wm_score() - curr_model.get_learning_rate() * err_wm_score)  # ws update
        err_h_curr = numpy.dot(curr_model.get_wm_score().T, err_score)  # error vector dht for deriving previous steps
        err_m_curr = 0          # last step has no next memory cell, dct = 0
        err_wm_hidden = 0       # last step has 0 accumulated weight error, dw = 0
    else:   # if not on last step, send ht, ct for forward prop, wait for dht, dct, dw for backward prop
        err_h_curr, err_m_curr, err_wm_hidden, loss = train(f_dataset_train[:, 1:], h_curr, m_curr, t_index)
    err_m_curr = err_m_curr + (err_h_curr * o_gate * (1 - numpy.power(numpy.tanh(m_curr), 2)))  # accumulative dct
    err_a_gate = err_m_curr * i_gate              # error vector dat calculated from dct
    err_i_gate = err_m_curr * a_gate              # error vector dat calculated from dct
    err_f_gate = err_m_curr * m_prev              # error vector dat calculated from dct
    err_o_gate = err_h_curr * numpy.tanh(m_curr)  # error vector dot calculated from dht
    err_m_prev = err_m_curr * f_gate              # error vector dct-1 calculated from dct
    # calculate error vector for raw, unsplit, unsquashed hidden values for all at, it, ft, and ot
    err_a_inp = err_a_gate * (1 - numpy.power(numpy.tanh(hidden_whole[:curr_model.get_hidden_unit_size(), :]), 2))
    err_i_inp = err_i_gate * i_gate * (1 - i_gate)
    err_f_inp = err_f_gate * f_gate * (1 - f_gate)
    err_o_inp = err_o_gate * o_gate * (1 - o_gate)
    err_inp_whole = numpy.vstack((err_a_inp, err_i_inp, err_f_inp, err_o_inp))   # dzt concated from hidden value errors
    err_wm_hidden = err_wm_hidden + numpy.dot(err_inp_whole, inp_whole.T)        # accumulative dw for weight update
    err_inp_unit = numpy.dot(curr_model.get_wm_hidden().T, err_inp_whole)        # whole input error vector dIt
    err_h_prev = err_inp_unit[input_size:, 0].reshape((curr_model.get_hidden_unit_size(), 1))    # dht-1 split from dIt
    return err_h_prev, err_m_prev, err_wm_hidden, loss      # returns dht-1, dct-1, dw for backprop


def validate(f_dataset_train, h_prev, m_prev, t_index):
    # do forward propagation similar to train function
    inp_unit = f_dataset_train[:, 0].reshape(input_size, 1)
    inp_whole = numpy.vstack((inp_unit, h_prev))
    hidden_whole = numpy.dot(curr_model.get_wm_hidden(), inp_whole)
    a_gate = numpy.tanh(hidden_whole[:curr_model.get_hidden_unit_size(), :])
    i_gate = expit(hidden_whole[curr_model.get_hidden_unit_size():2*curr_model.get_hidden_unit_size(), :])
    f_gate = expit(hidden_whole[2*curr_model.get_hidden_unit_size():3*curr_model.get_hidden_unit_size(), :] + f_bias)
    o_gate = expit(hidden_whole[3*curr_model.get_hidden_unit_size():, :])
    m_curr = i_gate * a_gate + f_gate * m_prev
    h_curr = o_gate * numpy.tanh(m_curr)
    if f_dataset_train[:, 1:].size == 0:    # on the last step, calculate P, E, no backpropagation
        score = numpy.dot(curr_model.get_wm_score(), h_curr)
        score_prob = numpy.exp(score) / numpy.sum(numpy.exp(score))
        loss = -numpy.log(score_prob[t_index])
        return loss
    else:   # if not on last step, continue forward propagation, pass on E
        loss = validate(f_dataset_train[:, 1:], h_curr, m_curr, t_index)
        return loss


def test(f_dataset_train, h_prev, m_prev):
    # do forward propagation similar to train function
    inp_unit = f_dataset_train[:, 0].reshape(input_size, 1)
    inp_whole = numpy.vstack((inp_unit, h_prev))
    hidden_whole = numpy.dot(curr_model.get_wm_hidden(), inp_whole)
    a_gate = numpy.tanh(hidden_whole[:curr_model.get_hidden_unit_size(), :])
    i_gate = expit(hidden_whole[curr_model.get_hidden_unit_size():2*curr_model.get_hidden_unit_size(), :])
    f_gate = expit(hidden_whole[2*curr_model.get_hidden_unit_size():3*curr_model.get_hidden_unit_size(), :] + f_bias)
    o_gate = expit(hidden_whole[3*curr_model.get_hidden_unit_size():, :])
    m_curr = i_gate * a_gate + f_gate * m_prev
    h_curr = o_gate * numpy.tanh(m_curr)
    if f_dataset_train[:, 1:].size == 0:    # on the last step, calculate P, no backpropagation
        score = numpy.dot(curr_model.get_wm_score(), h_curr)
        score_prob = numpy.exp(score) / numpy.sum(numpy.exp(score))
        return score_prob
    else:   # if not on last step, continue forward propagation, pass on P
        score_prob = test(f_dataset_train[:, 1:], h_curr, m_curr)
        return score_prob


print("LOADING VARIABLES...")


path_audio_raw = "raw"                 # directory storing raw spectrogram audio file
path_audio_converted = "converted"    # directory storing converted spectrogram image
path_saved_model = "model"             # directory storing LSTM model in pickles
audio_per_speaker = 0                   # number of sentences per speaker
uinp = 0                                # holds user input
speakers = []                           # array holding all speakers
audios = []                             # temporarily holds array of audios per speaker
learning_rate = 0.0         # adjustable parameter
hidden_unit_size = 0        # adjustable parameter
max_epoch = 3000            # fixed value parameter
input_size = 129            # fixed value parameter
f_bias = 5                  # fixed value parameter
target_array = []           # for accessing corresponding target matrix
decision_threshold = 0.5    # for accepting / rejecting a result
curr_model = Model(learning_rate, hidden_unit_size)     # default model initialization with 0 parameter value
main_menu = """
REGISTERED SPEAKERS: {}
REGISTERED AUDIOS: {}
AUDIO PER SPEAKER: {}

LEARNING RATE: {}
HIDDEN UNITS: {}
MAX EPOCH: {}

LSTM MODEL: {}
ACCURACY: {:.3f}%

1. Convert (converts audio from "raw/" into spectrogram image, and then into data, stored in "converted/")
2. Train (trains LSTM model with data from "converted/")
3. Test (tests LSTM model with data from "converted/")
4. Edit (edits parameters to optimize accuracy)
5. Save (saves LSTM model into pickle, stored in "model/")
6. Load (loads LSTM model from "model/")
0. Exit (exits the program)

Please enter a number."""


print("PREPARING DIRECTORIES...")


# ensure folder path exists
if not os.path.exists(path_audio_raw):
    os.mkdir(path_audio_raw)
if not os.path.exists(path_audio_converted):
    os.mkdir(path_audio_converted)
if not os.path.exists(path_saved_model):
    os.mkdir(path_saved_model)


print("\nWelcome to the LSTM Speaker Recognition program!")


# main program loop
while True:
    print(main_menu.format(Speaker.total_speakers, Audio.total_audios, audio_per_speaker,
                           curr_model.get_learning_rate(), curr_model.get_hidden_unit_size(), max_epoch,
                           curr_model.get_name(), curr_model.get_accuracy()))
    uinp = input("Choice: ")
    print("\n")
    if uinp == "1":
        # converts audio into spectrogram, then into numpy array
        print("CONVERTING AUDIO...\n")
        print("""LOADING AUDIO FILES FROM "raw/"...""")
        speakers = []               # empties speaker array
        target_array = []           # empties target array
        Speaker.total_speakers = 0  # initializes total speakers on reconvert
        Audio.total_audios = 0      # initializes total audios on reconvert
        for dr in os.listdir(path_audio_raw):   # for all DR folders
            dialect = dr                                # DR1 - DR8
            dr_path = os.path.join(path_audio_raw, dr)  # path into each DR directory
            for sp in os.listdir(dr_path):      # for all Speaker folders
                sex = sp[:1]                            # M || F
                speaker_id = sp[1:]                     # 3 characters for initial, with 1 additional number
                sp_path = os.path.join(dr_path, sp)     # path into each speaker directory
                audios = []                             # empties audio array
                for au in os.listdir(sp_path):  # for all .WAV files
                    sentence_type = au[:2]              # SA || SI || SX
                    sentence_number = au[2:].replace(".WAV", "")    # numbers from 1 to 4 digits
                    path_raw = os.path.join(sp_path, au)             # path to each audio
                    # path to each audio's spectrogram counterpart
                    path_converted = os.path.join(path_audio_converted, sp, au.replace(".WAV", ".png"))
                    # add audio object into audio array
                    audios.append(Audio(sentence_type, sentence_number, path_raw, path_converted))
                # add speaker object into speaker array including the respective audio array
                speakers.append(Speaker(dialect, sex, speaker_id, audios))
        if Speaker.total_speakers > 0:
            audio_per_speaker = round(Audio.total_audios / Speaker.total_speakers)      # calculate total dataset
            # create array the size of total speakers, each index containing respective target vectors
            for index, speaker in enumerate(speakers):
                temp_target = numpy.zeros((Speaker.total_speakers, 1), dtype=float)
                temp_target[index, 0] = 1
                target_array.append(temp_target)
                speaker.set_index(index)
                for audio in speaker.get_audios():
                    audio.set_index(index)

        print("""LOADING SUCCESSFUL.
        REGISTERED {} TOTAL SPEAKERS.
        REGISTERED {} TOTAL AUDIOS.
        WITH {} AUDIOS PER SPEAKER.\n""".format(Speaker.total_speakers, Audio.total_audios, audio_per_speaker))
        print("CONVERTING AUDIO INTO SPECTROGRAM IMAGES...")
        skipped = 0     # for keeping track of conversion status
        converted = 0
        err = 0
        for speaker in speakers:
            print("CONVERTING", speaker.get_name())
            # create path to each speaker's spectrogram directory
            if not os.path.exists(os.path.join(path_audio_converted, speaker.get_name())):
                os.mkdir(os.path.join(path_audio_converted, speaker.get_name()))
            for audio in speaker.get_audios():
                if os.path.exists(audio.get_path_converted()):     # skip if spectrogram file already exists
                    skipped += 1
                else:
                    # converts audio into spectrogram
                    # -n is necessary to generate the image, -Y sets max pixel height (rounded down to 2^n + 1),
                    # -X sets pps, -m monochrome, -r raw / no legend, -o output name
                    command = "sox {} -n spectrogram -Y 150 -X 50 -m -r -o {}".format(audio.get_path_raw(),
                                                                                      audio.get_path_converted())
                    # sends sox command into shell (run this program from shell!)
                    res = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    output, errors = res.communicate()     # returns result from subprocess
                    if errors:                             # detects any errors
                        err += 1
                        print("ERRORS:\n", errors)
                    else:                                  # counts successful conversions
                        converted += 1
        print("""CONVERSION SUCCESSFUL.
        {} AUDIO FILES SUCCESSFULLY CONVERTED.
        {} DUPLICATE AUDIO FILES SKIPPED.
        {} ERROR AUDIO FILES DETECTED.\n""".format(converted, skipped, err))
        print("CONVERTING SPECTROGRAM INTO NUMBERS...")
        # convert spectrogram image file into 2D array of numbers per pixel
        for speaker in speakers:
            for audio in speaker.get_audios():
                img = Image.open(audio.get_path_converted())    # opens image file with Pillow
                imgData = numpy.asarray(img)                    # saves image file as numpy array
                imgData = imgData/255                           # converts pixel color into 0 - 1 value
                audio.set_data(imgData)                         # adds numpy pixel array into the respective audio class
            random.shuffle(speaker.get_audios())                # initial shuffle to randomize audio order
        print("CONVERSION SUCCESSFUL.\n SPECTROGRAM IMAGES CONVERTED INTO NUMPY ARRAYS.\n")
        print("PRESS ENTER TO CONTINUE...")
        input()
    elif uinp == "2":
        # error handling
        if curr_model.get_hidden_unit_size() == 0 or Speaker.total_speakers == 0:
            print("PLEASE CONVERT AUDIOS AND INITIALIZE PARAMETERS FIRST")
            input("PRESS ENTER TO CONTINUE...")
            continue
        train_dataset = []              # array for training dataset
        val_dataset = []                # array for validate dataset
        curr_train_loss = 1000000000    # big number for loss initialization
        curr_val_loss = 1000000000
        # temporary weight matrix for storing best model performance, initialized safely as array
        temp_wmh = numpy.zeros((curr_model.get_hidden_unit_size()*4, input_size+curr_model.get_hidden_unit_size()))
        temp_wms = numpy.zeros((Speaker.total_speakers, curr_model.get_hidden_unit_size()))
        # create datasets
        for t_speaker in speakers:
            train_dataset_partial = [t_audio for t_audio in t_speaker.get_audios() if t_audio.get_usage() == "Train"]
            val_dataset_partial = [t_audio for t_audio in t_speaker.get_audios() if t_audio.get_usage() == "Validate"]
            train_dataset.extend(train_dataset_partial)
            val_dataset.extend(val_dataset_partial)
        print("START OF TRAINING: ", datetime.datetime.now())   # timestamps start of training
        for i in range(1, max_epoch + 1):  # 3000 epochs total
            print("EPOCH {}!".format(i))
            random.shuffle(train_dataset)   # shuffles train dataset for each epoch
            total_train_loss = 0            # initializes current train loss for observing
            for data in train_dataset:     # 6 audio for each speakers in train dataset
                print("#", end="")         # for tracking progress
                sys.stdout.flush()
                # does a forward and backward propagation for n timesteps, n the width of spectrogram
                err_wm, data_loss = train(data.get_data(),      # returns dw and loss for observing
                                          numpy.zeros((curr_model.get_hidden_unit_size(), 1), dtype=float),
                                          numpy.zeros((curr_model.get_hidden_unit_size(), 1), dtype=float),
                                          data.get_index())[2:]
                # updates model hidden weight matrix according to edw
                curr_model.set_wm_hidden(curr_model.get_wm_hidden() - curr_model.get_learning_rate() * err_wm)
                total_train_loss += data_loss     # accumulates train loss
            print("\n")
            train_loss = total_train_loss / len(train_dataset)      # averages train loss
            print("CURRTLOSS:\n", curr_train_loss)      # prints current best train loss
            print("TRAINLOSS:\n", train_loss)           # prints this step's train loss
            if curr_train_loss > train_loss:
                curr_train_loss = train_loss              # updates current best train loss
            if i % 5 == 0:                  # validate each 5 epochs
                total_val_loss = 0           # initializes validate loss
                for data in val_dataset:    # 2 audio for each speakers in validate dataset
                    print("#", end="")      # for tracking progress
                    sys.stdout.flush()
                    # does a forward propagation for n timesteps, returns validate loss
                    data_loss = validate(data.get_data(), numpy.zeros((
                        curr_model.get_hidden_unit_size(), 1), dtype=float),
                                         numpy.zeros((curr_model.get_hidden_unit_size(), 1), dtype=float),
                                         data.get_index())
                    total_val_loss = total_val_loss + data_loss     # accumulates validate loss
                print("\n")
                val_loss = total_val_loss / len(val_dataset)        # averages validate loss
                print("CURRVLOSS:\n", curr_val_loss)              # prints current best validate loss
                print("VALLOSS:\n", val_loss)                     # prints this step's validate loss
                if curr_val_loss > val_loss:
                    curr_val_loss = val_loss                        # updates curent best validate loss
                    temp_wmh = curr_model.get_wm_hidden()           # memorizes w at best validate loss
                    temp_wms = curr_model.get_wm_score()            # memorizes ws at best validate loss
        print("END OF TRAINING: ", datetime.datetime.now())      # timestamps end of training
        curr_model.set_wm_hidden(temp_wmh)      # at end of training, uses w with best validate loss
        curr_model.set_wm_score(temp_wms)       # at end of training, use ws with best validate loss
        print("TRAINING COMPLETED.")
        print("PRESS ENTER TO CONTINUE...")
        input()
    elif uinp == "3":
        # error handling
        if curr_model.get_hidden_unit_size() == 0 or Speaker.total_speakers == 0:
            print("PLEASE CONVERT AUDIOS AND INITIALIZE PARAMETERS FIRST")
            input("PRESS ENTER TO CONTINUE...")
            continue
        test_dataset = []   # initializes test dataset
        for t_speaker in speakers:  # create test dataset
            test_dataset_partial = [t_audio for t_audio in t_speaker.get_audios() if t_audio.get_usage() == "Test"]
            test_dataset.extend(test_dataset_partial)
        confusion_matrix = numpy.zeros((24, 24))    # create empty confusion matrix
        temp_accuracy = 0   # initializes test accuracy
        confidence = 0      # initializes test confidence
        print("START OF TEST: ", datetime.datetime.now())      # timestamps start of test
        for data in test_dataset:   # 2 audio for each speakers in test dataset
            # does a forward propagation for n timesteps, return score probability
            result = test(data.get_data(), numpy.zeros((
                        curr_model.get_hidden_unit_size(), 1), dtype=float),
                                         numpy.zeros((curr_model.get_hidden_unit_size(), 1), dtype=float))
            confidence += numpy.max(result)              # get the result's highest probability as confidence
            if numpy.max(result) < decision_threshold:   # if the confidence is less than 0.5, consider it a miss
                continue
            target = target_array[data.get_index()]      # gets the target vector
            temp_result = int(numpy.argmax(result))      # gets the index with highest probability (decision)
            temp_target = data.get_index()               # gets the index of actual target
            confusion_matrix[temp_target][temp_result] += 1     # updates confusion matrix accordingly
            if temp_target == temp_result:               # adds accuracy if decision and target matches
                temp_accuracy += 1
        curr_model.set_accuracy(temp_accuracy * 100 / len(test_dataset))    # updates the model's accuracy
        print("END OF TEST: ", datetime.datetime.now())      # timestamps end of test
        print("TEST COMPLETED ON {} AUDIOS".format(len(test_dataset)))
        print("ACCURACY: {:.3f}% ({} CORRECT PREDICTIONS)".format(curr_model.get_accuracy(), temp_accuracy))
        print("AVERAGE CONFIDENCE: {:.3f}%".format(confidence * 100 / len(test_dataset)))
        print("CONFUSION MATRIX:\n", confusion_matrix)
        print("PRESS ENTER TO CONTINUE...")
        input()
    elif uinp == "4":
        temp_lr = float(input("INPUT DESIRED LEARNING RATE: "))
        temp_hu = int(input("INPUT DESIRED HIDDEN UNIT SIZE: "))
        temp_uinp = input("THIS WILL DISCARD CURRENT LSTM MODEL. ARE YOU SURE YOU WANT TO CONTINUE (Y/N)?")
        if not (temp_uinp == "Y" or temp_uinp == "y"):  # confirmation check
            print("OPERATION CANCELLED.")
            print("PRESS ENTER TO CONTINUE...")
            input()
            continue
        else:
            print("CREATING MODEL...")
            learning_rate = temp_lr         # new learning rate
            hidden_unit_size = temp_hu      # new hidden unit size
            curr_model = Model(learning_rate, hidden_unit_size)     # initialize new model with new parameters
            print("MODEL SUCCESSFULLY INITIALIZED.")
            print("PRESS ENTER TO CONTINUE...")
            input()
    elif uinp == "5":
        # error handling
        if curr_model.get_hidden_unit_size() == 0 or Speaker.total_speakers == 0:
            print("PLEASE CONVERT AUDIOS AND INITIALIZE PARAMETERS FIRST")
            input("PRESS ENTER TO CONTINUE...")
            continue
        print("SAVING SPEAKER AND MODEL DATA...")
        save_data = speakers, curr_model    # save speakers and model data
        save_name = curr_model.get_name()   # create filename
        # handle single duplicate
        if os.path.exists(os.path.join(path_saved_model, curr_model.get_name())):
            print("DUPLICATE FILE FOUND. PLEASE RENAME MODELS BEFORE CREATING DUPLICATES.")
            save_name = save_name + "(1)"
        f = open(os.path.join(path_saved_model, save_name), "wb")   # open binary write file
        pickle.dump(save_data, f)       # saves data to a local file
        f.close()                       # close binary write file
        del save_data                   # clear memory used
        print("DATA SUCCESSFULLY SAVED AT PATH {} AS {}.".format(path_saved_model, curr_model.get_name()))
        print("PRESS ENTER TO CONTINUE...")
        input()
    elif uinp == "6":
        print("AVAILABLE MODELS:")
        for model in os.listdir(path_saved_model):  # prints models for ease of use
            print(model)
        print("Please enter the file name you wish to load.")
        load_name = input("Choice: ")
        if not os.path.exists(os.path.join(path_saved_model, load_name)):   # handle false inputs
            print("MODEL NOT FOUND.")
            print("PRESS ENTER TO CONTINUE...")
            input()
            continue
        print("LOADING SPEAKER AND MODEL DATA...")
        Speaker.total_speakers = 0      # initialize total speakers
        Audio.total_audios = 0          # initialize total audios
        f = open(os.path.join(path_saved_model, load_name), "rb")   # open binary read file
        load_data = pickle.load(f)      # loads data from local file
        speakers = load_data[0]         # assign speaker from loaded data
        curr_model = load_data[1]       # assign model from loaded data
        # update total speakers and total audios
        for speaker in speakers:
            Speaker.total_speakers = Speaker.total_speakers + 1
            for audio in speaker.get_audios():
                Audio.total_audios = Audio.total_audios + 1
        # create target array accordingly
        if Speaker.total_speakers > 0:
            audio_per_speaker = round(Audio.total_audios / Speaker.total_speakers)  # assign audio per speaker
            for index, speaker in enumerate(speakers):
                temp_target = numpy.zeros((Speaker.total_speakers, 1), dtype=float)
                temp_target[index, 0] = 1
                target_array.append(temp_target)
                speaker.set_index(index)
                for audio in speaker.get_audios():
                    audio.set_index(index)
        f.close()       # close binary read file
        del load_data   # clear memory
        print("DATA SUCCESSFULLY LOADED FROM PATH {} AS {}.\n".format(path_saved_model, load_name))
        print("PRESS ENTER TO CONTINUE...")
        input()
    elif uinp == "0":   # close program
        break
    else:
        # if input not detected correctly
        print("INPUT NOT RECOGNIZED.")
        print("PRESS ENTER TO CONTINUE...")
        input()
print("SHUTTING DOWN PROGRAM...")
