import datetime
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class Logger:
    """
    Singleton class of the logger system that will handle the logging system.
    ...

    Attributes
    ----------
    _logger_instance: Logger
        Represents the current running instance of Logger, this will only be created once (by default set to None).
    _print_statements_enabled : bool
        Represents if print statements are going to be enabled (by default set to False).
    """

    _logger_instance = None
    _print_statements_enabled = False

    @staticmethod
    def instance():
        """
        Obtains instance of Logger.
        """

        return Logger._logger_instance

    def __init__(self, _logger_dir: str) -> None:
        """
        Default constructor.
        """

        if Logger._logger_instance is None:
            Path(_logger_dir).mkdir(parents=True, exist_ok=True)

            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    TimedRotatingFileHandler(
                        os.path.join(_logger_dir, str(datetime.datetime.now().date()) + '-log.txt'),
                        when='midnight',
                        interval=1
                    ),
                    logging.StreamHandler()
                ]
            )
            Logger._logger_instance = self
        else:
            raise Exception("{}: Cannot construct, an instance is already running.".format(__file__))
