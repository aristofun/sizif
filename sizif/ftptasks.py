import ftplib
import logging
import socket
import sys
import time

"""
Atomic synchronous download/upload jobs to be executed either on main or background thread
"""

FTP_RETRY_COUNT = 7  # attempts to execute failed FTP operation
FTP_RETRY_DELAY = 5  # seconds between another FTP attempt

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
logger = logging.getLogger(__name__)


def download_task(from_filename, to_filepath, remote_folder='/',
                  host=None, port=0, login=None, password=None,
                  die_on_ftperrors=False,
                  verbose=0,
                  ftp=None):
    """
    Download with resume/reconnect support over ftplib.
    Synchronous IF to be wrapped in threads.

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

    :return True on successfull download, exception instance of last error on failed download
    """
    logger.debug(f'Downloading from {from_filename} to {to_filepath}')
    exception = None

    if ftp:
        dispose_ftp = False
        remote_folder = ftp.pwd()
    else:
        ftp = ftplib.FTP()
        ftp.set_debuglevel(verbose)
        ftp.set_pasv(True)
        dispose_ftp = True

    with open(to_filepath, 'w+b') as f:
        attempts = FTP_RETRY_COUNT

        # https://github.com/keepitsimple/pyFTPclient/blob/master/pyftpclient.py
        def connect():
            logger.debug(f'Connect to FTP: {host}:{port} {login}@***, dir={remote_folder}')
            ftp.connect(host, port)
            ftp.login(login, password)
            ftp.cwd(remote_folder)
            # optimize socket params for download task
            ftp.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            ftp.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 75)
            ftp.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)

        connect()
        target_size = ftp.size(from_filename)
        logger.debug(f'Target download size: {target_size}')

        while target_size > f.tell():
            try:
                connect()
                # retrieve file from position where we were disconnected
                if f.tell() == 0:
                    ftp.retrbinary(f'RETR {from_filename}', f.write)
                else:
                    ftp.retrbinary(f'RETR {from_filename}', f.write, rest=f.tell())
            except ftplib.all_errors as e:
                logger.exception(exc_info=e)
                attempts -= 1

                if attempts < 1:
                    logger.error(
                        f'{FTP_RETRY_COUNT} download attempts made, f.tell() is {f.tell()}')
                    if die_on_ftperrors:
                        raise e
                    else:
                        exception = e
                        break

                logger.debug(f'Waiting {FTP_RETRY_DELAY} sec before download resume...')
                time.sleep(FTP_RETRY_DELAY)

            finally:
                if dispose_ftp:
                    ftp.close()
                    logger.debug(f'FTP closed.')
            logger.debug(f'Downloaded {f.tell()}/{target_size} bytes')

    if exception:
        return exception
    else:
        return True


def upload_task(from_filepath, to_filename, remote_folder='/',
                host=None, port=0, login=None, password=None,
                die_on_ftperrors=False,
                verbose=0,
                ftp=None):
    pass
