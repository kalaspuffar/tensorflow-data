# tensorflow-data
An small example how to use tensorflow data (tf.data)

### Usage

In order to try this repository you clone it on your drive. You'll probably need 12-20Gb of disk because of the large amount of image data.

First of download the image [Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765) and put the picture in a structure where you have
```
./PetImages/Cat/*.jpg
./PetImages/Dog/*.jpg
```


I've not pin-point everything required to run this test scripts because I had most of it installed already.
But you need to install tensorflow, opencv2 and numpy atleast.
```
pip install tensorflow opencv2-python numpy
```

Then you run the create_dataset.py in order to create the train, test and validation tfrecords.
```
python create_dataset.py
```

Lastly you can train your model using the training script.
```
python train.py
```

If you have any question or suggestion the just reach out. Open an issue and I'll look into it.
