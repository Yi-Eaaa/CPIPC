import logging
import os
import time


class Logger:
    def __init__(self, log_name="default", log_dir="./logs"):
        self.log_dir = log_dir
        self.log_name = log_name
        self._setup_logger()

    def _setup_logger(self):
        os.makedirs(self.log_dir, exist_ok=True)

        log_file = os.path.join(
            self.log_dir, f'{self.log_name}_{time.strftime("%Y%m%d_%H%M%S")}.log'
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(self.log_name)

    def get_logger(self):
        return self.logger
