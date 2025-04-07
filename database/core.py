from datetime import datetime
from enum import Enum

from sqlalchemy import Column, Integer, DateTime, ForeignKey, VARCHAR, Boolean, Table
from sqlalchemy.orm import relationship

from database.database import Base, engine


train_dataset = Table(
    "train_dataset",
    Base.metadata,
    Column("train_id", Integer, ForeignKey("train.id"), primary_key=True),
    Column("dataset_id", Integer, ForeignKey("dataset.id"), primary_key=True)
)


class Notation(Base):
    __tablename__ = "notation"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)

    labels = relationship("Label", back_populates="notation")


class Label(Base):
    __tablename__ = "label"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(VARCHAR(128), nullable=False)
    description = Column(VARCHAR(255), nullable=True)
    dataset_id = Column(Integer, ForeignKey("dataset.id"))
    notation_id = Column(Integer, ForeignKey("notation.id"))
    user_id = Column(Integer, ForeignKey('user.id'), unique=False)

    dataset = relationship("Dataset", back_populates="labels")
    notation = relationship("Notation", back_populates="labels")


class Train(Base):
    __tablename__ = "train"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_date = Column(DateTime, default=datetime.utcnow(), nullable=True)

    name = Column(VARCHAR(128))
    status = Column(VARCHAR(255))
    job_id = Column(VARCHAR(255))
    sample_rate = Column(Integer, default=8000)
    num_samples = Column(Integer, default=22050)
    fold = Column(Integer, default=5)
    epochs = Column(Integer, default=50)
    batch_size = Column(Integer, default=128)

    user_id = Column(Integer, ForeignKey('user.id'), unique=False)

    datasets = relationship("Dataset", secondary=train_dataset, back_populates="trains")


class Dataset(Base):
    __tablename__ = "dataset"

    id = Column(Integer, primary_key=True, autoincrement=True)

    created_date = Column(DateTime, default=datetime.utcnow(), nullable=True)
    updated_date = Column(DateTime, default=datetime.utcnow(), onupdate=datetime.utcnow(), nullable=True)

    country = Column(VARCHAR(100), nullable=True)

    user_id = Column(Integer, ForeignKey('user.id'), unique=False)

    trains = relationship("Train", secondary=train_dataset, back_populates="datasets")
    labels = relationship("Label", back_populates="dataset")


class UserRole(Base):
    __tablename__ = "user_role"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(VARCHAR(128), nullable=True)

    users = relationship('User', back_populates='role')


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_date = Column(DateTime, default=datetime.utcnow(), nullable=True)
    updated_date = Column(DateTime, default=datetime.utcnow(), onupdate=datetime.utcnow(), nullable=True)
    first_name = Column(VARCHAR(128), nullable=True)
    last_name = Column(VARCHAR(128), nullable=True)
    email = Column(VARCHAR(255), nullable=True)
    phone = Column(VARCHAR(128), nullable=True)
    username = Column(VARCHAR(128), unique=True, nullable=True)
    password = Column(VARCHAR(255), nullable=True)
    role_id = Column(Integer, ForeignKey('user_role.id'))

    role = relationship('UserRole', back_populates='users')


class AudioFile(Base):
    __tablename__ = "audio_file"

    id = Column(Integer, primary_key=True, autoincrement=True)
    extension = Column(VARCHAR(255), nullable=True)

    label_id = Column(Integer, ForeignKey("label.id"))
    dataset_id = Column(Integer, ForeignKey("dataset.id"))
    notation_id = Column(Integer, ForeignKey("notation.id"))

    # label = relationship("Label", back_populates="audio_files")


class TrainStatus(Enum):
    RUNNING = 1
    QUEUED = 2
    COMPLETED = 3
    FAILED = 4


Base.metadata.create_all(engine)
