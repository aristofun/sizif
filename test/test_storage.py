import json
import os
import shutil
from unittest import TestCase

from sizif.storage import FileCheckpointsMonitor


def rel(file_path):
    return os.path.join(os.path.dirname(__file__), file_path)


class TestFileStorage(TestCase):

    # @classmethod
    # def setUpClass(cls):
    #     cls.agender = PyAgender()

    def test_attributes(self):
        fm = FileCheckpointsMonitor(folder='./votka//', version='24/',
                                    file_template='/weights/{epoch:04d}-{'':.3f}.hdf5',
                                    rotate_number=-1,
                                    verbose=1)
        self.assertEqual(fm.current_checkpoint, '')
        self.assertEqual(fm.current_params, {'checkpoint': ''})
        self.assertEqual(fm.checkpoints, [])
        self.assertEqual(fm.rotate_number, -1)
        self.assertEqual(fm.verbose, 1)
        self.assertEqual(fm.state_filepath, 'votka/currentstate_24.json')
        self.assertEqual(fm.checkpoint_path, 'votka/model_24__weights_{epoch:04d}-{'':.3f}.hdf5')
        self.assertTrue(os.path.exists(fm.state_filepath))
        os.remove(fm.state_filepath)
        os.rmdir(os.path.dirname(fm.state_filepath))

    def test_writing_checkpoints_raise_error(self):
        fm = FileCheckpointsMonitor(version=0, file_template='hey.txt')
        self.assertEqual(fm.checkpoints, [])
        with self.assertRaises(FileNotFoundError):
            fm.on_checkpoint_written(fm.checkpoint_path, {})

        # cleanup
        shutil.rmtree(os.path.dirname(fm.state_filepath))

    def test_rotate_reset(self):
        fm = FileCheckpointsMonitor(version=1, file_template='he{o}y', rotate_number=3)
        for i in range(6):
            fname = fm.checkpoint_path + str(i)
            open(fname, 'w').close()
            fm.on_checkpoint_written(fname, {})

        self.assertEqual(3, len(fm.checkpoints))
        self.assertEqual(fm.current_checkpoint, fm.checkpoint_path + '5')
        self.assertFalse(os.path.exists(fm.checkpoint_path + '0'))
        self.assertFalse(os.path.exists(fm.checkpoint_path + '1'))
        self.assertFalse(os.path.exists(fm.checkpoint_path + '2'))

        self.assertTrue(os.path.exists(fm.checkpoint_path + '3'))
        fm.rotate_checkpoints(2)
        self.assertFalse(os.path.exists(fm.checkpoint_path + '3'))
        self.assertEqual(2, len(fm.checkpoints))

        with open(fm.state_filepath, "r") as fp:
            data = json.load(fp)
        self.assertEqual(data['checkpoint'], fm.current_checkpoint)

        fm.reset()
        self.assertEqual([], fm.checkpoints)

        with open(fm.state_filepath, "r") as fp:
            data = json.load(fp)

        self.assertEqual('', fm.current_checkpoint)
        self.assertEqual(data['checkpoint'], fm.current_checkpoint)

        # cleanup
        shutil.rmtree(os.path.dirname(fm.state_filepath))

    def test_writing_checkpoints_autorotate(self):
        fm = FileCheckpointsMonitor(version=2, file_template='hey.txt', rotate_number=3)
        self.assertEqual([], fm.checkpoints)

        for i in range(2):
            fname = fm.checkpoint_path + str(i)
            open(fname, 'w').close()
            fm.on_checkpoint_written(fname, {})

        self.assertEqual(2, len(fm.checkpoints))
        self.assertEqual(fm.current_checkpoint, fm.checkpoint_path + '1')

        for i in range(2, 5):
            fname = fm.checkpoint_path + str(i)
            open(fname, 'w').close()
            fm.on_checkpoint_written(fname, {})

        self.assertEqual(3, len(fm.checkpoints))
        self.assertEqual(fm.current_checkpoint, fm.checkpoint_path + '4')
        self.assertFalse(os.path.exists(fm.checkpoint_path + '0'))
        self.assertFalse(os.path.exists(fm.checkpoint_path + '1'))

        # cleanup
        shutil.rmtree(os.path.dirname(fm.state_filepath))
