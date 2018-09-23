from keras.callbacks import Callback
from keras.engine.saving import load_model
from keras.engine.training import Model
import numpy as np


class KerasModelWrapper:
    """
    Keras model lifecycle manager. Runs training, loads and saves model states via provided storage.

    Instance variables:
         model — keras model instance being managed, must be compiled
         storage - storage manager being used for data i/o

    All calls except Keras's :py:meth:`Model.fit` and :py:meth:`Model.fit_generator` are
    proxied to the model 'as is'

    :py:meth:`Model.fit` and :py:meth:`Model.fit_generator` are wrapped with model save/restore
    operations
    """

    def __init__(self, model, storage):
        """
        :param model: compiled and ready to use keras model
        :param storage: sizif storage manager instance with all parameter for lifecycle control
        """
        self.storage = storage
        self.model = model
        self.callback = KerasModelWrapper.KerasCallback(self.storage)

        if not isinstance(self.model, Model):
            raise ValueError('model_fn must provide instance of keras.engine.training.Model')

    def fit(self, *args, restart_storage=False, **kwargs):
        """
        Proxy all *args and **kwargs to keras model `fit` sugarcoated with state recovery and backup

        :param restart_storage:
                True — reset storage stats and monitoring, not resetting the model itself
                False — always load `storage.current_checkpoint` to the model before fitting
        """
        if restart_storage:
            self.storage.reset()
            print(f'\nModel has been reset before fitting\n')
        else:
            self.load_model()

        # injecting variables
        kwargs.setdefault('callbacks', []).append(self.callback)
        kwargs['initial_epoch'] = self.storage.current_params.get('epoch', 0)  # saved epoch

        self.model.fit(*args, **kwargs)

    def fit_generator(self, *args, restart_storage=False, **kwargs):
        """
        Proxy all *args and **kwargs to keras model `fit_generator` sugarcoated with state recovery
        and backup

        :param restart_storage:
                True — reset storage stats and monitoring, not resetting the model itself
                False — always load `storage.current_checkpoint` to the model before fitting
        """
        if restart_storage:
            self.storage.reset()
            print(f'\nModel has been reset before fitting\n')
        else:
            self.load_model()

        # injecting variables
        kwargs.setdefault('callbacks', []).append(self.callback)
        kwargs['initial_epoch'] = self.storage.current_params.get('epoch', 0)  # saved epoch

        self.model.fit_generator(*args, **kwargs)

    def load_model(self):
        """
        Explicitly load recent model state from `storage.current_checkpoint` if present
        """
        if self.storage.current_checkpoint:
            if self.storage.save_weights_only:
                self.model.load_weights(self.storage.current_checkpoint)
            else:
                self.model = load_model(self.storage.current_checkpoint)
            print(f'\nModel weights loaded from: {self.storage.current_checkpoint}')
        else:
            print(f'\nModel weights not loaded. cp: "{self.storage.current_checkpoint}"')

    def __getattr__(self, name):
        """
        Wildcard method/attribute proxy to the model.
        For experimental use, use direct .model object for real stuff.
        """
        return getattr(self.model, name)

    class KerasCallback(Callback):
        """
        Middleman receiving notifications from keras model and notifying storage monitor

        Almost identical to Keras ModelCheckpoint, cause model snapshots management is based
        on local filesystem.

        All paramaters except `self.model_wrapper` are as of Keras ModelCheckpoint
        """

        def __init__(self, storage):
            self.storage = storage

            # -------- Keras code -------------
            super().__init__()
            self.monitor = storage.monitor
            self.verbose = storage.verbose
            self.filepath = storage.checkpoint_path
            self.save_best_only = storage.save_best_only
            self.save_weights_only = storage.save_weights_only
            self.period = storage.period
            self.epochs_since_last_save = 0

            mode = storage.mode

            if mode not in ['auto', 'min', 'max']:
                print(f'ModelCheckpoint mode {mode} is unknown, fallback to auto mode.')
                mode = 'auto'
            if mode == 'min':
                self.monitor_op = np.less
                self.best = np.Inf
            elif mode == 'max':
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                    self.monitor_op = np.greater
                    self.best = -np.Inf
                else:
                    self.monitor_op = np.less
                    self.best = np.Inf

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            filepath = self.__do_save_model(epoch, logs)

            if filepath:  # notify storage
                ep = {'epoch': epoch + 1}
                self.storage.on_checkpoint_written(filepath, {**logs, **ep})

        def __do_save_model(self, epoch, logs):
            """
            :return: actual filepath of written model, None if nothing was actually saved to disk
            """
            self.epochs_since_last_save += 1
            if self.epochs_since_last_save >= self.period:
                self.epochs_since_last_save = 0
                filepath = self.filepath.format(epoch=epoch + 1, **logs)

                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        print(
                            f'\nCan save best model only with {self.monitor} available, skipping.')
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s'
                                      % (epoch + 1, self.monitor, self.best,
                                         current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                            return filepath
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)
                    return filepath
