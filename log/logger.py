import logging
import time
import datetime


class RuntimeFormatter(logging.Formatter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()

    def formatTime(self, record, datefmt=None):
        duration = datetime.datetime.utcfromtimestamp(record.created - self.start_time)
        elapsed = duration.strftime('%H:%M:%S')
        return str(elapsed)


class MyLogger:

    def __init__(self, base_logger):
        self.base_logger = base_logger
        self.history = []

    def __call__(self, msg, is_print=True):
        self.history.append(msg)
        if is_print:
            print(msg)
        self.base_logger.info(msg)

    def write(self, writer):
        writer.add_text(f"Logging", '<br />'.join(self.history), 0)
        return writer


def get_logger(filepath=None):
    lg = logging.getLogger("Mylogger")
    lg.setLevel(logging.INFO)
    if filepath:
        file = logging.FileHandler(filepath, mode='w')
        file.setFormatter(RuntimeFormatter('%(asctime)s - %(message)s'))
        lg.addHandler(file)
    return MyLogger(lg)


logger = get_logger()

if __name__ == '__main__':
    """ quick test """

    logger = get_logger()
    logger("Program started")
    time.sleep(1)
    logger("kek", is_print=True)
