# Deep learning models lifecycle management backup/restore nano framework
Currently only for Keras >= 2.2 models.

<image src="./pic.png" />

# How to use

```commandline
pip3 install sizif
```

Local filesystem Keras checkpoints backup: 

```python
from sizif.keras import KerasModelWrapper
from sizif.storage import FileCheckpointsMonitor

# your compiled Keras Model instance
model = build_model()  


# Snapshots monitor
# Different model architectures should have different version parameter
# other parameters similar to Keras ModelCheckpoint
cpm = FileCheckpointsMonitor(version=1,
                            file_template='weights.{epoch:03d}-vl{val_loss:.3f}.hdf5',
                            folder='./checkpoints',
                            rotate_number=5,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=False,
                            save_weights_only=True,
                            mode='auto',
                            period=1)

# Keras wrapper, proxies all calls to the model
# except fit and fit_generator — which are surrounded 
# by automated model state backup/recovery   
km = KerasModelWrapper(model, cpm)

# all method parameters ar proxied to Keras as is except callbacks
km.fit_generator(training_set_generator,
                 epochs=25,
                 validation_data=test_set_generator,
                 callbacks=[tboard])
``` 

See sources for detailed docstrings

## TODO: 
- FTP/S3/SFTP/Dropbox uploading monitors
- Tensorflow/Pytorch models support

## Tests

```commandline
python3 -m unittest 
```

## Dependencies
- numpy ~> 1.15
- Keras ~> 2.2

## License

This project is released under the MIT license.