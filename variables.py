import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Variables:
    """ This class is responsible for saving and
        loading variables from the system environment.

        Initiated at the entry point
    """
    admin_default_password: str = os.getenv(
        "USER_DEFAULT_PASSWORD",
        "password"
    )
    mariadb_database_user: str = os.getenv(
        "MARIADB_DATABASE_USER",
        "root"
    )
    mariadb_database_password: str = os.getenv(
        "MARIADB_DATABASE_PASSWORD",
        "root"
    )
    mariadb_database_host: str = os.getenv(
        "MARIADB_DATABASE_HOST",
        "127.0.0.1"
    )
    mariadb_database_port: int = int(os.getenv(
        "MARIADB_DATABASE_PORT",
        3306
    ))
    mariadb_database_name: str = os.getenv(
        "MARIADB_DATABASE_NAME",
        "amd_train"
    )
    app_host: str = os.getenv(
        "APP_HOST",
        "127.0.0.1"
    )
    app_port: int = int(os.getenv(
        "APP_PORT",
        8001
    ))

    logger_dir: str = os.getenv(
        "LOGGER_DIR",
        "/stor/data/logs/amd-train/"
    )
    file_dir: str = os.getenv(
        "FILE_DIR",
        "/stor/data/amd-train/"
    )
    base_dir = os.path.dirname(__file__)


variables = Variables()
