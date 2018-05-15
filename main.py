print("IMPORTING PACKAGES...")

import os                   #for directories
import time                 #for naming files
import pickle               #for storing objects
import subprocess           #for sending shell commands
import numpy                #for matrix calculations
from PIL import Image       #for converting grayscale image into numbers

class Speaker:
    #holds information about registered speakers
    total_speakers = 0
    def __init__(self, s_dialect, s_sex, s_id, s_audios):
        self.dialect = s_dialect
        self.sex = s_sex
        self.speaker_id = s_id
        self.audios = s_audios
        Speaker.total_speakers += 1
        #refreshes audio usage distribution per speaker
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

class Audio:
    #holds individual audio, each speaker has 10 of these as list
    total_audios = 0
    total_validate = 0
    total_train = 0
    total_test = 0
    total_validate_SI = 0
    total_validate_SX = 0
    total_test_SI = 0
    total_test_SX = 0

    def __init__(self, a_type, a_number, a_path_raw, a_path_converted):

        #using core test set, 24 speaker, each 10 sentence, 2 SA, 3 SI, 5 SX
        #2 validation, 6 train, 2 test
        #1 SI 1 SX validation, 2 SA 1 SI 3 SX train, 1 SI 1 SX test

        #The dialect sentences (SA). To expose dialectal difference. 2 sentences, read by all speakers.
        #The phonetically-compact sentences (SX). To slightly differentiate phonetic context. 5 sentences per speaker, 1 speakers per sentence.
        #The phonetically-diverse sentences (SI). To diversify phonetic context. 3 sentences per speaker, 1 speaker per sentence.

        self.sentence_type = a_type
        self.sentence_number = a_number
        self.path_raw = a_path_raw
        self.path_converted = a_path_converted
        Audio.total_audios += 1
        #distribute usage between audios according to predetermined values
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

    def set_data(self, a_data):
        self.data = a_data

    def get_data(self):
        return self.data

#prepare the directories used
print("BUILDING DIRECTORIES...")
#folder paths
audio_raw = "raw"                 #raw spectrogram audio file
audio_converted = "converted"    #pickle storing converted spectrogram image
saved_model = "model"                   #pickle storing LSTM model

if not os.path.exists(audio_raw):               #ensures folder exists
    os.mkdir(audio_raw)
if not os.path.exists(audio_converted):         #ensures folder exists
    os.mkdir(audio_converted)
if not os.path.exists(saved_model):             #ensures folder exists
    os.mkdir(saved_model)

#loads main loop
print("PREPARING MAIN LOOP...")
audio_per_speaker = 0
menu_table = \
"""
\nREGISTERED SPEAKERS: {}
REGISTERED AUDIOS: {}
AUDIO PER SPEAKER: {}

LEARNING RATE: x
HIDDEN UNITS: y
MAX EPOCH: z

LSTM MODEL: NULL
ACCURACY: NULL

1. Convert (converts audio from "raw/" into spectrogram image, and then into data, stored in "converted/")
2. Train (trains LSTM model with data from "converted/")
3. Test (tests LSTM model with data from "converted/")
4. Edit (edits parameters to optimize accuracy)
5. Save (saves LSTM model into pickle, stored in "model/")
6. Load (loads LSTM model from "model/")
0. Exit (exits the program)

Please enter a number."""

print("\nWelcome to the LSTM Speaker Recognition program!\n")
while True:
    #main loop
    print(menu_table.format(Speaker.total_speakers, Audio.total_audios, audio_per_speaker))
    choice = input("Choice: ")
    print("\n")
    if choice == "1":
        #converts audio into spectrogram, and then into numpy array, and store it as classes
        print("CONVERTING AUDIO...\n")

        print("""LOADING AUDIO FILES FROM "raw/"...""")

        speakers = []
        for dr in os.listdir(audio_raw):
            dialect = dr
            dr_path = os.path.join(audio_raw, dr)
            for sp in os.listdir(dr_path):
                if len(sp) == 0: break
                sex = sp[:1]
                speaker_id = sp[1:]
                sp_path = os.path.join(dr_path, sp)
                audios = []
                for au in os.listdir(sp_path):
                    if len(au) == 0: break
                    sentence_type = au[:2]
                    sentence_number = au[2:].replace(".WAV", "")
                    path_raw = os.path.join(sp_path, au)
                    path_converted = os.path.join(audio_converted, sp, au.replace(".WAV", ".png"))
                    audios.append(Audio(sentence_type, sentence_number, path_raw, path_converted))
                speakers.append(Speaker(dialect, sex, speaker_id, audios))
        # for speaker in speakers:
        #     print(speaker.get_dialect(), speaker.get_sex(), speaker.get_id())
        #     for audio in speaker.get_audios():
        #         print(audio.get_usage(), audio.get_type(), audio.get_number(), audio.get_path_raw(), audio.get_path_converted())

        if(Speaker.total_speakers > 0): audio_per_speaker = round(Audio.total_audios / Speaker.total_speakers)

        print("""LOADING SUCCESSFUL.
        REGISTERED {} TOTAL SPEAKERS.
        REGISTERED {} TOTAL AUDIOS.
        WITH {} AUDIOS PER SPEAKER.\n""".format(Speaker.total_speakers, Audio.total_audios, audio_per_speaker))

        input("Press enter to continue...")

        print("CONVERTING AUDIO INTO SPECTROGRAM IMAGES...")

        skipped = 0
        converted = 0
        err = 0
        for speaker in speakers:
            print("CONVERTING", speaker.get_name())
            if (not os.path.exists(os.path.join(audio_converted, speaker.get_name()))):
                os.mkdir(os.path.join(audio_converted, speaker.get_name()))
            for audio in speaker.get_audios():
                if(os.path.exists(audio.get_path_converted())):
                    skipped += 1
                else:
                    #-n is necessary to generate the image, -Y sets max pixel height (rounded down to 2^n + 1), -X sets pps, -m monochrome, -r raw / no legend, -o output name
                    command = "sox {} -n spectrogram -Y 150 -X 50 -m -r -o {}".format(audio.get_path_raw(), audio.get_path_converted())
                    res = subprocess.Popen(command, shell=True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
                    output, errors = res.communicate()
                    if(errors):
                        err += 1
                        print("ERRORS:\n", errors)
                    else:
                        converted += 1
                        #print("OUTPUT:\n", output)

        print("""CONVERSION SUCCESSFUL.
        {} AUDIO FILES SUCCESSFULLY CONVERTED.
        {} DUPLICATE AUDIO FILES SKIPPED.
        {} ERROR AUDIO FILES DETECTED.\n""".format(converted, skipped, err))

        input("Press enter to continue...")

        print("CONVERTING SPECTROGRAM INTO NUMBERS...")

        converted = 0
        for speaker in speakers:
            for audio in speaker.get_audios():
                print(audio.get_path_converted())
                img = Image.open(audio.get_path_converted())
                width, height = img.size
                imgData = numpy.asarray(img)
                print("{}x{}\n{}".format(width, height, imgData))
                imgData = imgData/255
                audio.set_data(imgData)
                converted += 1

        print("""CONVERSION SUCCESSFUL.
        {} SPECTROGRAM IMAGES CONVERTED INTO NUMPY ARRAYS.\n""".format(converted))

        input("Press enter to continue...")


    elif choice == "0":
        break
    else:
        print("Input not recognized")
        continue

print("Shutting down program...")

# close necessary objects

print("Shut down")
