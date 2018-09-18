import json
import os
import re


class FileCheckpointsMonitor:
    """
    Local filesystem interface for counting and rotating saved model snapshots.

    NOTE: Snapshots rotation is in the scope of current class instance lifecycle.
          New object will not rotate files from previous run.
          Unless snapshots file names are the same

    Public read-only properties:
        current_checkpoint - local path to recently written checkpoint file
        current_params - dict with additional params of currently written checkpoint
        checkpoints - list of all checkpoint file names available to current instance
        rotate_number - number of recent checkpoints to keep, -1 meaning never rotate
        state_file - local path to status file

        checkpoint_path - checkpoints file path template to use in code actually saving checkpoints

        All other options are equivalent to :py:class:`Keras ModelCheckpoint`
    """

    def __init__(self, version, file_template, folder='./checkpoints', rotate_number=5,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 period=1
                 ):
        """
        Setup files, folders and instance variables.

        :param folder: where to save current status file and all weights snapshots,
                must be writable by the process.

        :param version: number or short string unique for given set of file snapshots
                (i.e. DL model architecture).
                If not unique - weights from different model would be loaded or failed.
                Only numbers or alphanumeric characters and _ . allowed.

        :param file_template: filename to save checkpoints to inside folder

        :param rotate_number: number of recent checkpoints to keep, -1 or 0 meaning keep all.
        
        :param verbose: 0 - silent, 1 - print out key operations to STDOUT.

        :param monitor: quantity to monitor.

        :param save_best_only: if `save_best_only=True`,
                the latest best model according to
                the quantity monitored will not be overwritten.

        :param mode: one of {auto, min, max}.
                If `save_best_only=True`, the decision
                to overwrite the current save file is made
                based on either the maximization or the
                minimization of the monitored quantity. For `val_acc`,
                this should be `max`, for `val_loss` this should
                be `min`, etc. In `auto` mode, the direction is
                automatically inferred from the name of the monitored quantity.

        :param save_weights_only: if True, then only the model's weights will be
                saved (`model.save_weights(filepath)`), else the full model
                is saved (`model.save(filepath)`).

        :param period: Interval (number of epochs) between checkpoints.

        """
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.monitor = monitor
        self.period = period

        self.verbose = int(verbose)
        self.rotate_number = int(rotate_number) or -1

        model_version = re.sub(r'[^\w.]', '', str(version))
        status_filename = f'currentstate_{model_version}.json'
        self.state_file = os.path.normpath(os.path.join(folder, status_filename))

        file_name = 'model_' + model_version + '_' + str(file_template).strip('/\\')
        # '.weights.{epoch:04d}-{' + monitor + ':.3f}.hdf5'

        self.checkpoint_path = os.path.normpath(os.path.join(folder, file_name))
        self.checkpoints = list()
        self.current_checkpoint = ''
        self.current_params = dict()

        # https://stackoverflow.com/a/12517490/1245302
        # check folder access and initialize current status file value
        if not os.path.isfile(self.state_file):
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            self.reset()
        else:
            self._load_current_status()

    # ---------------- PUBLIC interface --------------------------------------------

    def on_checkpoint_written(self, file_path, params):
        """
        Must be called after actual file is written to local FS. Makes storage aware and
        able to run additional processing.

        :param file_path:
        :param params: additional params dict to save to `state_file` like epoch number, val_loss etc.
        :raise: FileNotFoundError if file_path doesn't exist or not accessible
        """
        self._verbose(f'#on_checkpoint_written({file_path}, ...)')
        params = params or {}
        # if checkpoint is not present
        if not (os.path.isfile(file_path) and os.access(file_path, os.R_OK)):
            raise FileNotFoundError(f'"{file_path}" file is not available!')

        # update current_checkpoint, current_params
        self._add_checkpoint(file_path, params)
        self._save_current_status()

        self.rotate_checkpoints()
        self._verbose(f'checkpoint processed successfully')

    def rotate_checkpoints(self, rotate_number=None):
        """
        Removes all checkpoint files except recent `rotate_number`

        :param rotate_number: self.rotate_number is used if 0 or None provided
                -1 meaning no rotation
        :return: True if any actual checkpoints were removed, False otherwise
        """
        rotate_number = int(rotate_number or self.rotate_number)  # that's right, no 0 rotation!
        if rotate_number < 1: return False

        old_size = len(self.checkpoints)
        to_remove = self.checkpoints[:-rotate_number]
        self.checkpoints = self.checkpoints[-rotate_number:]
        new_size = len(self.checkpoints)

        for i in to_remove:
            try:
                os.remove(i)
            except OSError as e:
                print(f'cant remove file: {e}')

        self._verbose(f'rotated! Checkpoints size: {new_size}')

        return bool(new_size - old_size)

    def reset(self):
        """
        Clears all checkpoints and saves blank checkpoint status file. No checkpoints are rotated.
        """
        self._verbose(f'#reset() checkpoints size: {len(self.checkpoints)}')
        self.current_checkpoint = ''
        self.checkpoints.clear()
        self._save_current_status()

    # --------------- protected methods --------------------------------------------

    def _load_current_status(self, reset_on_deadcheckpoint=True):
        with open(self.state_file, "r") as fp:
            data = json.load(fp)

        self._add_checkpoint(data['checkpoint'], data)

        # if checkpoint is not present - reset the checkpoint
        if reset_on_deadcheckpoint \
                and not (os.path.isfile(self.current_checkpoint)
                         and os.access(self.current_checkpoint, os.R_OK)):
            self.reset()

    def _save_current_status(self):
        self.current_params['checkpoint'] = self.current_checkpoint

        with open(self.state_file, "w") as fp:
            json.dump(self.current_params, fp)

    def _add_checkpoint(self, cp, params):
        self.current_params = params
        self.current_checkpoint = cp
        if self.current_checkpoint: self.checkpoints.append(self.current_checkpoint)

    def _verbose(self, string):
        if self.verbose > 0: print('FileCheckpointsMonitor: '+string)
