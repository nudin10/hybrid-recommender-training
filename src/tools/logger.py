import logging

class Logger:
    def __init__(self, name: str = "Runpod", level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt='[%(levelname)s][%(asctime)s] %(name)s : %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # prevents log duplication
        self.logger.propagate = False

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)

global_logger: Logger = None #type:ignore

def init_global_logger(level=logging.INFO) -> Logger:
    global global_logger
    if global_logger is None:
        global_logger = Logger(level=level)
    return global_logger

def get_global_logger() -> Logger:
    if global_logger is None:
        raise RuntimeError("Logger global_logget not initialised. Call init_global_logger() to initialise.")
    return global_logger
