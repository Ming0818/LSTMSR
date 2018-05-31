![alt tag](https://github.com/xaltt/LSTMSR/blob/master/img/chuckles.png)

# LSTMSR

Speaker Recognition with low-level Long Short-Term Memory for assignment

Trains a manually-coded LSTM model to identify human voice using [TIMIT corpus](https://github.com/philipperemy/timit) as dataset and optimizes accuracy by tweaking learning rate and number of units


Requirements:

```
python
sox
```

Additional packages:

```
numpy
pillow
scipy
```

- Place raw audio files in raw/
- Converted audio files are stored in converted/
- Save/load pickles of trained LSTM models in model/


External links:

[Python 3.6.4](https://www.python.org/downloads/release/python-364/)

[SoX (Sound eXchange)](https://sourceforge.net/projects/sox/files/sox/)

[NumPy](https://pypi.org/project/numpy/)

[Pillow](https://pypi.org/project/Pillow/2.2.1/)

[SciPy](https://pypi.org/project/scipy/)
