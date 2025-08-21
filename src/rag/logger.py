import logging
import os
import time


class Logger:
    def __init__(self, log_name="default", log_dir="./logs"):
        self.log_dir = log_dir
        self.log_name = log_name
        self.logger = self._setup_logger()

    def _setup_logger(self):
        os.makedirs(self.log_dir, exist_ok=True)

        log_file = os.path.join(
            self.log_dir, f'{self.log_name}_{time.strftime("%Y%m%d_%H%M%S")}.log'
        )

        logger = logging.getLogger(self.log_name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            stream_handler = logging.StreamHandler()

            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

        return logger

    def get_logger(self):
        return self.logger
