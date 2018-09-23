
<image src="https://github.com/aristofun/sizif/raw/master/pic.png" align="right" width=240 />

# DL backup/restore nano framework

Auto backup/restore model snapshots of deep learning models:

- to/from local filesystem
- to/from remote FTP server

Current version supports only Keras >= 2.2 models. You're welcome to contribute.


# Usage

```commandline
pip3 install sizif
```

FTP Keras checkpoints backup/restore: 

```python
from sizif.keras import KerasModelWrapper
from sizif.storage import FTPFileCheckpointsMonitor

# your compiled Keras Model instance
model = build_model()  


# Local filesystem snapshots monitor with FTP backup/restore 
# Different model architectures should have different version parameter
# other parameters similar to Keras ModelCheckpoint
# See sizif.storage.FileCheckpointsMonitor for local file only backup/restore 
cpm = FTPFileCheckpointsMonitor(1,
                               'weights.{epoch:03d}-vl{val_loss:.3f}-va{val_acc:.3f}.hdf5',
                               local_folder='/snapshots_local_dir',
                               remote_folder='/snapshots_ftp_dir',
                               host='ftp.your-host.com', login='your_ftp_login',
                               password='your_ftp_password',
                               die_on_ftperrors=True,
                               rotate_number=3,
                               monitor='val_loss',
                               verbose=1,
                               save_best_only=False,
                               save_weights_only=True,
                               mode='auto',
                               period=1)
                               
# Keras wrapper, proxies all calls to the model
# except `fit` and `fit_generator` — which are surrounded 
# by automated model state backup/recovery   
km = KerasModelWrapper(model, cpm)

# all method parameters are proxied to Keras as is except callbacks
# callbacks are extended with `cpm` listener 
km.fit_generator(training_set_generator,
                 epochs=25,
                 validation_data=test_set_generator,
                 callbacks=[tboard])
``` 

See sources for detailed docstrings

## TODO: 
- SSH/S3/Dropbox uploading monitors
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
