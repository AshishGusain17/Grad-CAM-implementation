# Grad-CAM-implementation

## Dependencies required

* Python 3.0
* TensorFlow 2.0
* tf-explain
* keras


```
Grad-CAM uses the gradient information flowing into any particular convolutional layer to produce a coarse 
localization map highlighting the important regions in the image to understand each neuron for a decision of interest.
```

## gradCAM implementation

```
First one is the normal image and the second one has gradCAM implemented over its last convolutional layer.
The third one has highlighted regions displayed over the initial image.
```
<img src="https://github.com/AshishGusain17/Grad-CAM-implementation/blob/master/display/resnet50.png?raw=true" >



<br />

### Working with tf-explain
```
This library is not an official Google product, although it is built for Tensorflow 2.0 models.
tf-explain is completely built over the new TF2.0 API, and is primarily based on tf.keras when possible.
Implementing it over the classical dance images shows the results as:
```

<img src="https://github.com/AshishGusain17/Grad-CAM-implementation/blob/master/display/tf_explain.png?raw=true" >



