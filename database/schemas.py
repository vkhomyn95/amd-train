from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


class NotationSchema(BaseModel):
    id: Optional[int] = None
    name: str

    class Config:
        from_attributes = True
        extra = 'ignore'


class LabelSchema(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    dataset_id: int  # Required
    notation_id: int  # Required
    user_id: int  # Required
    notation: Optional["NotationSchema"]
    dataset: Optional["DatasetSchema"]

    class Config:
        from_attributes = True
        extra = 'ignore'


class TrainSchema(BaseModel):
    id: Optional[int]
    created_date: Optional[datetime] = None
    name: Optional[str]
    sample_rate: Optional[int]
    num_samples: Optional[int]
    epochs: Optional[int]
    batch_size: Optional[int]
    user_id: Optional[int]

    datasets: Optional[List["DatasetSchema"]] = []

    class Config:
        from_attributes = True
        extra = 'ignore'


class DatasetSchema(BaseModel):
    id: Optional[int]
    created_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
    country: Optional[str]
    user_id: Optional[int]

    trains: Optional[List[TrainSchema]] = []

    class Config:
        from_attributes = True
        extra = 'ignore'


class UserRoleSchema(BaseModel):
    id: Optional[int]
    name: Optional[str]

    class Config:
        from_attributes = True


class UserSchema(BaseModel):
    id: Optional[int] = None
    created_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    role_id: Optional[int] = None

    role: Optional[UserRoleSchema] = None

    class Config:
        from_attributes = True
        extra = 'ignore'
