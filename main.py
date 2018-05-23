print("INITIALIZING PROGRAM...")


import os                   # for directories
import time                 # for naming files
import pickle               # for storing objects
import subprocess           # for sending shell commands
import numpy                # for matrix calculations
from PIL import Image       # for converting gray scale image into numbers


class Model:                # for storing LSTM models
    def __init__(self, m_learning_rate, m_hidden_unit_size):
        self.learning_rate = m_learning_rate     # adjustable parameter
        self.hidden_unit_size = m_hidden_unit_size         # adjustable parameter
        self.accuracy = 0                        # tested parameter
        self.name = "lr{}h{}".format(self.learning_rate, self.hidden_unit_size)  # for identifying models
        # 4 gates, each a dxn matrix to handle input, and each with a dxd matrix for recursion
        #       at     wc uc                d = hidden_unit_size    n = input_size
        # zt = [it] = [wi ui] x [ xt ]      a, i, f, o = gate value
        #       ft     wf uf     ht-1       w, u = weight matrices
        #       ot     wo uo                x = input               h = recursion value
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

    def get_accuracy(self):
        return self.accuracy

    def get_wm_hidden(self):
        return self.wm_hidden

    def get_wm_score(self):
        return self.wm_score


class Speaker:              # for storing speaker data
    total_speakers = 0

    def __init__(self, s_dialect, s_sex, s_id, s_audios):
        self.dialect = s_dialect
        self.sex = s_sex
        self.speaker_id = s_id
        self.audios = s_audios
        self.index = None
        Speaker.total_speakers += 1
        # refreshes audio usage distribution per speaker
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
    total_audios = 0
    total_validate = 0
    total_train = 0
    total_test = 0
    total_validate_SI = 0   # for usage ratio
    total_validate_SX = 0
    total_test_SI = 0
    total_test_SX = 0

    def __init__(self, a_type, a_number, a_path_raw, a_path_converted):
        self.sentence_type = a_type
        self.sentence_number = a_number
        self.path_raw = a_path_raw
        self.path_converted = a_path_converted
        self.index = None
        self.data = None
        Audio.total_audios += 1
        if self.sentence_type == "SX" and Audio.total_validate_SX < 1:
            self.usage = "Validate"
            Audio.total_validate_SX += 1
            Audio.total_validate += 1
        elif self.sentence_type == "SX" and Audio.total_test_SX < 1:
            self.usage = "Test"
            Audio.total_test_SX += 1
            Audio.total_test += 1
        elif self.sentence_type == "SI" and Audio.total_validate_SI < 1:
            self.usage = "Validate"
            Audio.total_validate_SI += 1
            Audio.total_validate += 1
        elif self.sentence_type == "SI" and Audio.total_test_SI < 1:
            self.usage = "Test"
            Audio.total_test_SI += 1
            Audio.total_test += 1
        else:
            self.usage = "Train"
            Audio.total_train += 1

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


def sigmoid(num):
    return 1 / (1 + numpy.exp(-num))


def train():
    train_dataset = []
    val_dataset = []
    for t_speaker in speakers:
        train_dataset_partial = [t_audio for t_audio in t_speaker.get_audios() if t_audio.get_usage() == "Train"]
        val_dataset_partial = [t_audio for t_audio in t_speaker.get_audios() if t_audio.get_usage() == "Validate"]
        train_dataset.extend(train_dataset_partial)
        val_dataset.extend(val_dataset_partial)
    # for data in train_dataset:
    #     print(data.get_data().shape, data.get_usage())
    #     print(data.get_data()[:,0].reshape(129, 1))
    #     print(data.get_data()[:,1:].shape)
    #     print(numpy.zeros((129, 1)).shape)
    #     print(numpy.vstack((data.get_data()[:,0].reshape(129, 1), numpy.zeros((129, 1)))))
    # for data in val_dataset:
    #     print(data.get_data().shape, data.get_usage())
    for i in range(1, max_epoch):  # 3000 epochs
        print("EPOCH {}!".format(i))
        if i % 5 == 5:          # validate each 5 epochs
            print("VALIDATE TIME!")
        for data in train_dataset:
            backprop(data.get_data(), numpy.zeros((curr_model.get_hidden_unit_size(), 1), dtype=float),
                     numpy.zeros((curr_model.get_hidden_unit_size(), 1), dtype=float), data.get_index())


def backprop(f_dataset_train, h_prev, m_prev, t_index):
    inp_unit = f_dataset_train[:, 0].reshape(input_size, 1)
    inp_whole = numpy.vstack((inp_unit, h_prev))
    hidden_whole = numpy.dot(curr_model.get_wm_hidden(), inp_whole)
    a_gate = numpy.tanh(hidden_whole[:hidden_unit_size, :])
    i_gate = sigmoid(hidden_whole[hidden_unit_size:2*hidden_unit_size, :])
    f_gate = sigmoid(hidden_whole[2*hidden_unit_size:3*hidden_unit_size, :] + f_bias)
    o_gate = sigmoid(hidden_whole[3*hidden_unit_size:, :])
    m_curr = i_gate * a_gate + f_gate * m_prev
    h_curr = o_gate * numpy.tanh(m_curr)
    if f_dataset_train[:, 1:].size != 0:
        backprop(f_dataset_train[:, 1:], h_curr, m_curr, t_index)
    else:
        score = numpy.dot(curr_model.get_wm_score(), h_curr)
        score_prob = numpy.exp(score) / numpy.sum(numpy.exp(score))
        # a straightforward solution to calculate J
        # jac = numpy.zeros((len(score), len(score_prob)), dtype=float)
        # for i in range(len(score)):
        #     for j in range(len(score_prob)):
        #         if i == j:
        #             jac[i, j] = score_prob[i] * (1 - score_prob[j])
        #         else:
        #             jac[i, j] = -score_prob[i] * score_prob[j]
        # and a concise, one liner solution to calculate J
        jac = numpy.diagflat(score_prob) - numpy.dot(score_prob, score_prob.T)  # J = yi x (1{i = j} - yj)
        err_score = numpy.dot(jac, score_prob - target_array[t_index])          # errs = J . (y - t)
        err_wm_score = numpy.dot(err_score, h_curr.T)                           # errws = errs . hcurr_T
        err_h_curr = numpy.dot(curr_model.get_wm_score().T, err_score)          # errhc = ws_T . errs



print("LOADING VARIABLES...")


path_audio_raw = "raw"                 # raw spectrogram audio file
path_audio_converted = "converted"    # pickle storing converted spectrogram image
path_saved_model = "model"             # pickle storing LSTM model
audio_per_speaker = 0                   # number of sentences per speaker
uinp = 0                                # holds user input
speakers = []                           # array holding all speakers
audios = []                                # temporarily holds array of audios per speaker
learning_rate = 0.0  # adjustable parameter
hidden_unit_size = 0    # adjustable parameter
max_epoch = 3000   # fixed value parameter
input_size = 129   # fixed value parameter
f_bias = 5         # fixed value parameter
target_array = []  # for accessing corresponding target matrix
curr_model = Model(learning_rate, hidden_unit_size)
main_menu = """
REGISTERED SPEAKERS: {}
REGISTERED AUDIOS: {}
AUDIO PER SPEAKER: {}

LEARNING RATE: {}
HIDDEN UNITS: {}
MAX EPOCH: {}

LSTM MODEL: {}
ACCURACY: {}

1. Convert (converts audio from "raw/" into spectrogram image, and then into data, stored in "converted/")
2. Train (trains LSTM model with data from "converted/")
3. Test (tests LSTM model with data from "converted/")
4. Edit (edits parameters to optimize accuracy)
5. Save (saves LSTM model into pickle, stored in "model/")
6. Load (loads LSTM model from "model/")
0. Exit (exits the program)

Please enter a number."""


print("PREPARING...")


if not os.path.exists(path_audio_raw):               # ensures folder exists
    os.mkdir(path_audio_raw)
if not os.path.exists(path_audio_converted):         # ensures folder exists
    os.mkdir(path_audio_converted)
if not os.path.exists(path_saved_model):             # ensures folder exists
    os.mkdir(path_saved_model)


print("\nWelcome to the LSTM Speaker Recognition program!")


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
        speaker = []    # empties speaker array
        Speaker.total_speakers = 0
        for dr in os.listdir(path_audio_raw):
            dialect = dr                                # DR1 - DR8
            dr_path = os.path.join(path_audio_raw, dr)  # path into each DR directory
            for sp in os.listdir(dr_path):
                sex = sp[:1]                            # M || F
                speaker_id = sp[1:]                     # 3 characters for initial, with 1 additional number
                sp_path = os.path.join(dr_path, sp)     # path into each speaker directory
                audios = []                             # empties audio array
                for au in os.listdir(sp_path):
                    sentence_type = au[:2]              # SA || SI || SX
                    sentence_number = au[2:].replace(".WAV", "")    # numbers from 1 to 4 digits
                    path_raw = os.path.join(sp_path, au)             # path to each audio
                    # path to each audio's spectrogram counterpart
                    path_converted = os.path.join(path_audio_converted, sp, au.replace(".WAV", ".png"))
                    # add audio object into the array
                    audios.append(Audio(sentence_type, sentence_number, path_raw, path_converted))
                # add speaker object into the array including audios array
                speakers.append(Speaker(dialect, sex, speaker_id, audios))
        # count total audio per speaker
        target_array = []   # empties target array
        if Speaker.total_speakers > 0:
            audio_per_speaker = round(Audio.total_audios / Speaker.total_speakers)
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
        input("Press enter to continue...")
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
                    output, errors = res.communicate()      # returns result from subprocess
                    if errors:                             # detects any detected errors
                        err += 1
                        print("ERRORS:\n", errors)
                    else:                                   # counts successful conversions
                        converted += 1
                        # print("OUTPUT:\n", output)
        print("""CONVERSION SUCCESSFUL.
        {} AUDIO FILES SUCCESSFULLY CONVERTED.
        {} DUPLICATE AUDIO FILES SKIPPED.
        {} ERROR AUDIO FILES DETECTED.\n""".format(converted, skipped, err))
        input("Press enter to continue...")
        print("CONVERTING SPECTROGRAM INTO NUMBERS...")
        for speaker in speakers:
            for audio in speaker.get_audios():
                img = Image.open(audio.get_path_converted())    # opens image file with Pillow
                imgData = numpy.asarray(img)                    # saves image file as numpy array
                imgData = imgData/255                           # converts pixel color into 0 - 1 value
                audio.set_data(imgData)                         # adds numpy pixel array into the respective audio class
        print("CONVERSION SUCCESSFUL.\n SPECTROGRAM IMAGES CONVERTED INTO NUMPY ARRAYS.\n")
        input("Press enter to continue...")
    elif uinp == "2":
        train()
    elif uinp == "4":
        temp_lr = float(input("INPUT DESIRED LEARNING RATE: "))
        temp_hu = int(input("INPUT DESIRED HIDDEN UNIT SIZE: "))
        temp_uinp = input("THIS WILL DISCARD CURRENT LSTM MODEL. ARE YOU SURE YOU WANT TO CONTINUE (Y/N)?")
        if not (temp_uinp == "Y" or temp_uinp == "y"):
            continue
        else:
            learning_rate = temp_lr
            hidden_unit_size = temp_hu
            curr_model = Model(learning_rate, hidden_unit_size)
    elif uinp == "0":
        break
    else:
        # if input not detected correctly
        print("Input not recognized")
        continue


print("SHUTTING DOWN PROGRAM...")
