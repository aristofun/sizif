import ftplib
import logging
import os
import platform
import socket
import time

"""
Atomic synchronous download/upload jobs to be executed either on main or background thread
"""

FTP_RETRY_COUNT = 10  # attempts to execute failed FTP operation
FTP_RETRY_DELAY = 3  # seconds between another FTP attempt
FTP_OPERATION_TIMEOUT = 70.  # seconds per single blocking socket operation
MACOS = 'darwin' in platform.system().lower()  # socket options different for macOS and Linux

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# lh = logging.StreamHandler(sys.stdout)
# lh.setLevel(logging.DEBUG)
# logger.addHandler(lh)
logger.setLevel(logging.DEBUG)


def download_task(from_filename, to_filepath, remote_folder='/',
                  host=None, port=0, login=None, password=None,
                  die_on_ftperrors=False,
                  verbose=0,
                  ftp=None):
    """
    Download with resume/reconnect support over ftplib.
    Local file will be rewritten.
    Synchronous IF to be wrapped in threads.
    If verbose > 0, file progress %% are printed to stdout

    :param from_filename: remote filename within remote_folder to download
    :param to_filepath: local path to download to
    :param remote_folder: remote folder to do ftp.cwd() before downloading
    :param host: remote host
    :param port: remote port
    :param login: FTP login
    :param password: FTP password
    :param die_on_ftperrors: True — reraise ftplib errors, False - just log and return last exception
    :param verbose: ftplib verbosity option
    :param ftp: FTP() instance to operate on, if None - new created and closed after download complete

    :return (True, None) on successfull download, (False, exception) of last error on failed download
            Exception is raised immediately if die_on_ftperrors == True
    """
    logger.info(f'Downloading from {from_filename} to {to_filepath}')
    ftpw = FTPWrapper(remote_folder, host, port, login, password, verbose, ftp)
    exception = None

    with open(to_filepath, 'w+b') as f:
        attempts = FTP_RETRY_COUNT
        downloading = True
        logger.debug(f'Downloading {from_filename} to {to_filepath}...')

        while downloading:
            try:
                ftpw.connect()
                ftpw.ftp.voidcmd('TYPE I')
                target_size = ftpw.ftp.size(from_filename)
                tracker = ProgressTracker(target_size, f.tell(), verbose)

                def write_progress(block):
                    f.write(block)
                    tracker.handle(block)

                # retrieve file from position where we were disconnected
                # Possible issue with big files:
                # https://stackoverflow.com/questions/19692739/python-ftplib-hangs-at-end-of-transfer
                if f.tell() == 0:
                    ftpw.ftp.retrbinary(f'RETR {from_filename}', write_progress)
                else:
                    ftpw.ftp.retrbinary(f'RETR {from_filename}', write_progress, rest=f.tell())

                downloading = False
                logger.info(f'Downloaded {f.tell()}/{target_size} bytes to {to_filepath}')
            except ftplib.all_errors as e:
                logger.exception(f'Error downloading {from_filename}', exc_info=e)
                attempts -= 1

                if attempts < 1:
                    logger.error(
                        f'{FTP_RETRY_COUNT} attempts made, {from_filename}.tell() is {f.tell()}')
                    if die_on_ftperrors:
                        ftpw.close()
                        raise e
                    else:
                        exception = e
                        break

                logger.debug(f'Waiting {FTP_RETRY_DELAY} sec before download resume...')
                time.sleep(FTP_RETRY_DELAY)

            ftpw.close()

    if exception:
        return (False, exception)
    else:
        return (True, None)


def upload_task(from_filepath, to_filename, remote_folder='/',
                host=None, port=0, login=None, password=None,
                die_on_ftperrors=False,
                verbose=0,
                ftp=None):
    """
    Upload with resume/reconnect support over ftplib. Remote file will be rewritten!
    Synchronous IF to be wrapped in threads.
    If verbose > 0, file progress % printed to stdout

    Params are equivalent to :py:meth:`download_task`

    :return (True, None) on successfull download, (False, exception) of last error on failed download
            Exception is raised immediately if die_on_ftperrors == True
    """
    logger.info(f'Uploading from {from_filepath} to {remote_folder}/{to_filename}')
    ftpw = FTPWrapper(remote_folder, host, port, login, password, verbose, ftp)
    exception = None
    first_access = True

    # https://stackoverflow.com/questions/42019279/resumable-file-upload-in-python
    with open(from_filepath, 'rb') as f:
        attempts = FTP_RETRY_COUNT
        target_size = os.path.getsize(from_filepath)
        uploading = True
        tracker = None

        logger.debug(f'File of size {target_size} opened for upload...')

        while uploading:
            try:
                ftpw.connect()

                rest_pos = 0
                # Get file size if exists
                file_list = ftpw.ftp.nlst()

                # file was already there - need rewrite
                if to_filename in file_list:
                    if first_access:
                        logger.debug(f'{to_filename} is on server, will be rewritten')
                    else:
                        ftpw.ftp.voidcmd('TYPE I')
                        rest_pos = ftpw.ftp.size(to_filename)
                        logger.debug(
                            f'{to_filename} exists on server, resumed from {rest_pos} byte')

                f.seek(rest_pos, 0)
                tracker = ProgressTracker(target_size, rest_pos, verbose)
                first_access = False
                ftpw.ftp.storbinary('STOR ' + to_filename, f, callback=tracker.handle,
                                    rest=(rest_pos or None))

                uploading = False
                logger.info(
                    f'Uploaded {tracker.bytes_processed}/{target_size} bytes from {from_filepath}')
            except ftplib.all_errors as e:
                logger.exception(f'Error uploading {from_filepath}', exc_info=e)
                attempts -= 1

                if attempts < 1:
                    logger.error(
                        f'{FTP_RETRY_COUNT} attempts made'
                        + (f'{tracker.bytes_processed} bytes uploaded' if tracker else ''))
                    if die_on_ftperrors:
                        ftpw.close()
                        raise e
                    else:
                        exception = e
                        break

                logger.debug(f'Waiting {FTP_RETRY_DELAY} sec before upload resume...')
                time.sleep(FTP_RETRY_DELAY)
            ftpw.close()

    if exception:
        return (False, exception)
    else:
        return (True, None)


class FTPWrapper:
    """
    ftplib.FTP wrapper for reconnect/close operations
    """

    def __init__(self, remote_folder='/',
                 host=None, port=0, login=None, password=None,
                 verbose=0, ftp=None):
        self.remote_folder = remote_folder
        self.host = host
        self.port = port
        self.login = login
        self.password = password
        self.verbose = verbose

        if ftp:
            self.remote_folder = ftp.pwd()
            self.host = ftp.host
            self.port = ftp.port
            self.dispose_ftp = False
        else:
            ftp = ftplib.FTP(timeout=FTP_OPERATION_TIMEOUT)
            ftp.set_debuglevel(verbose)
            self.dispose_ftp = True

        self.ftp = ftp
        self.ftp.set_pasv(True)

        # https://github.com/keepitsimple/pyFTPclient/blob/master/pyftpclient.py

    def connect(self):
        logger.debug(
            f'Connect to FTP: {self.host}:{self.port} {self.login}@***, dir={self.remote_folder}')
        self.ftp.connect(self.host, self.port)
        self.ftp.login(self.login, self.password)
        self.ftp.cwd(self.remote_folder)
        # optimize socket params for download task
        self.ftp.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # https://stackoverflow.com/questions/12248132/how-to-change-tcp-keepalive-timer-using-python-script
        if MACOS:
            tcp_keepalive = 0x10
            self.ftp.sock.setsockopt(socket.IPPROTO_TCP, tcp_keepalive, 50)
        else:
            self.ftp.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 40)
            self.ftp.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 50)

    def close(self):
        if self.dispose_ftp:
            self.ftp.close()
            logger.debug(f'FTP closed.')


class ProgressTracker:
    def __init__(self, total_size, offset=0, verbose=0):
        self.verbose = verbose
        self.total_size = total_size
        self.bytes_processed = offset
        self.percent_complete = round(self.bytes_processed / self.total_size * 100)

    def handle(self, block):
        self.bytes_processed += len(block)
        self.percent_complete = round(self.bytes_processed / self.total_size * 100)

        if self.verbose > 0:
            print(f'\r{self.percent_complete}%', flush=True, end='\r')
            if self.complete():
                print(f'100% complete ({self.bytes_processed} bytes processed)')

    def complete(self):
        return self.bytes_processed >= self.total_size
