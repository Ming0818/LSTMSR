print("Importing packages...")

import os       #for directories
import time     #for naming files
import pickle   #for storing objects


print("Building directories...")
#folder paths
spectrogram_raw = "raw"                 #raw spectrogram audio file
spectrogram_converted = "converted"    #pickle storing converted spectrogram data
saved_model = "model"                   #pickle storing LSTM model

if not os.path.exists(spectrogram_raw):         #ensures folder exists
    os.mkdir(spectrogram_raw)
if not os.path.exists(spectrogram_converted):   #ensures folder exists
    os.mkdir(spectrogram_converted)
if not os.path.exists(saved_model):             #ensures folder exists
    os.mkdir(saved_model)

print("Preparing main loop...")
menu_table = \
"""1. Convert (converts audio from "raw/" into spectrogram image, and then into data, stored in "converted/")
2. Train (trains LSTM model with data from "converted/")
3. Test (tests LSTM model with data from "converted/")
4. Edit (edits parameters to optimize accuracy)
5. Save (saves LSTM model into pickle, stored in "model/")
6. Load (loads LSTM model from "model/")
0. Exit (exits the program)

Please enter a number."""

print("Welcome to the LSTM Speaker Recognition program!\n")
while True:
    #main loop
    print(menu_table)
    choice = input("Choice: ")
    if choice == "1":
        print("Converting audio...")

        print("""Loading audio from "raw/"...""")

        print("Converting audio into spectrogram images...")

        print("Converting spectrogram images into numbers...")

        print("""Storing numbers as pickle in "converted/"...""")

        print("""Conversion successful.
        X audio files successfully converted.
        Y duplicate names detected and skipped.\n""")

        input("Press enter to continue...")


    elif choice == "0":
        break
    else:
        print("Input not recognized")
        continue

print("Shutting down program...")

# close necessary objects

print("Shut down")
