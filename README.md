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


## Tensorflow lite model.

In order to use this model on a tensorflow lite enabled device you need to freeze your model using this command

```
freeze_graph \
  --input_graph=./model2/graph.pbtxt \
  --input_checkpoint=./model2/model.ckpt-81852 \
  --input_binary=false \
  --output_graph=/tmp/frozen.pb \
  --output_node_names=input_tensor,output_pred
```

In order to convert your model you need a tool called toco (Tensorflow Lite Optimizing Converter). Use the command below to build this tool in your tensorflow directory.
```
bazel build //tensorflow/contrib/lite/toco:toco
```

After that you convert it into a tensorflow lite model using the command below inside of your tensorflow directory.

```
 ./bazel-bin/tensorflow/contrib/lite/toco/toco 
   --input_file=/tmp/frozen.pb 
   --input_format=TENSORFLOW_GRAPHDEF 
   --output_format=TFLITE 
   --output_file=/tmp/cat_vs_dogs.tflite 
   --input_arrays=input_tensor 
   --output_arrays=output_pred 
   --input_shapes=1,224,224,3
```

Modifying the demo provided by google you can then test your inference on your device.