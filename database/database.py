from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from variables import variables

engine = create_engine(
    f"mariadb://"
    f"{variables.mariadb_database_user}:"
    f"{variables.mariadb_database_password}@"
    f"{variables.mariadb_database_host}:"
    f"{variables.mariadb_database_port}/"
    f"{variables.mariadb_database_name}"
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
