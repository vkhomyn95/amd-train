import logging
from typing import List

from fastapi import HTTPException
from sqlalchemy.orm import Session
from werkzeug.security import generate_password_hash

from database import UserRole, User, Dataset, Train
from database.core import Notation, Label, AudioFile
from database.schemas import UserSchema, DatasetSchema, LabelSchema
from variables import variables


def insert_default_roles(db: Session) -> None:
    try:
        default_roles = ['admin', 'supervisor', 'guest']

        for role_name in default_roles:
            if not db.query(UserRole).filter(UserRole.name == role_name).first():
                role = UserRole(name=role_name)
                db.add(role)

        db.commit()
    except Exception as e:
        db.rollback()
        print(f">>>Error inserting default roles: {e}")
    finally:
        db.close()


def insert_default_notation(db: Session) -> None:
    try:
        default_roles = ['human', 'voicemail', 'ring']

        for role_name in default_roles:
            if not db.query(Notation).filter(Notation.name == role_name).first():
                notation = Notation(name=role_name)
                db.add(notation)

        db.commit()
    except Exception as e:
        db.rollback()
        print(f">>>Error inserting default notations: {e}")
    finally:
        db.close()


def insert_default_user(db: Session) -> UserSchema:
    try:
        if not db.query(User).filter(User.username == "admin").first():
            default_user = User(
                username="admin",
                password=generate_password_hash(variables.admin_default_password),
                first_name="Administrator",
                last_name="Voiptime",
                email="support@voiptime.net",
                role_id=1
            )
            db.add(default_user)
            db.commit()
            return UserSchema.from_orm(default_user)
    except Exception as e:
        db.rollback()
        print(f">>>Error inserting default user: {e}")
    finally:
        db.close()


def insert_user(db: Session, user: User) -> UserSchema:
    try:
        db.add(user)
        db.commit()
        return UserSchema.from_orm(user)
    except Exception as e:
        db.rollback()
        print(f">>>Error inserting default user: {e}")
    finally:
        db.close()


def insert_dataset(db: Session, dst: Dataset) -> DatasetSchema:
    try:
        db.add(dst)
        db.commit()
        db.refresh(dst)
        return DatasetSchema.from_orm(dst)
    except Exception as e:
        db.rollback()
        print(f">>>Error inserting default dataset: {e}")
    finally:
        db.close()


def insert_label(db: Session, dst: Label):
    try:
        db.add(dst)
        db.commit()
        db.refresh(dst)
        return dst
    except Exception as e:
        db.rollback()
        print(f">>>Error inserting labels dataset: {e}")
    finally:
        db.close()


def insert_train(db: Session, train: Train, dataset_ids: List[int]):
    db.add(train)
    for dataset_id in dataset_ids:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset:
            train.datasets.append(dataset)
    db.commit()
    db.refresh(train)
    return train


def update_train(db: Session, train: Train):
    db.add(train)
    db.commit()
    db.refresh(train)
    return train


def load_user_by_api_key(db: Session, api_key: str):
    try:
        return db.query(User).filter(User.api_key == api_key).first()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_user_by_username(db: Session, username: str, email: str) -> UserSchema:
    try:
        user = db.query(User).filter((User.username == username) | (User.email == email)).first()
        return UserSchema.from_orm(user)
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_user_by_id(db: Session, user_id: int):
    try:
        return db.query(User).filter(User.id == user_id).first()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_train_by_id(db: Session, t_id: int):
    try:
        return db.query(Train).filter(Train.id == t_id).first()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_user_by_uuid(db: Session, user_uuid: str):
    try:
        return db.query(User).filter(User.uuid == user_uuid).first()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_users(db: Session, limit: int, offset: int, current_user_id: int):
    try:
        return db.query(User).filter(User.id != current_user_id).limit(limit).offset(offset).all()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def count_users(db: Session):
    try:
        return db.query(User).count()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return 0


def load_dataset_by_id(db: Session, dataset_id: int):
    try:
        return db.query(Dataset).filter(Dataset.id == dataset_id).first()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_dataset_by_id_related_to_user(db: Session, dataset_id: int, user_id: int):
    try:
        return db.query(Dataset).filter(Dataset.id == dataset_id).filter(Dataset.user_id == user_id).first()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_label_by_id(db: Session, dataset_id: int):
    try:
        return db.query(Label).filter(Label.id == dataset_id).first()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_dataset_by_country_user_id(db: Session, dataset_id: str, user_id: int):
    try:
        return db.query(Dataset).filter(Dataset.country == dataset_id).filter(Dataset.user_id == user_id).first()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_dataset_by_country(db: Session, dataset_id: str):
    try:
        return db.query(Dataset).filter(Dataset.country == dataset_id).first()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_datasets(db: Session, limit: int, offset: int):
    try:
        return db.query(Dataset).limit(limit).offset(offset).all()
    except Exception as e:
        logging.error(f'  >> Error during query:t {e}')
        db.rollback()
        return None


def load_datasets_related_to_user(db: Session, user_id: int, limit: int, offset: int):
    try:
        return db.query(Dataset).filter(Dataset.user_id == user_id).limit(limit).offset(offset).all()
    except Exception as e:
        logging.error(f'  >> Error during query:t {e}')
        db.rollback()
        return None


def load_datasets_all(db: Session):
    try:
        return db.query(Dataset).all()
    except Exception as e:
        logging.error(f'  >> Error during query:t {e}')
        db.rollback()
        return None


def load_datasets_all_related_to_user(db: Session, user_id: int):
    try:
        return db.query(Dataset).filter(Dataset.user_id == user_id).all()
    except Exception as e:
        logging.error(f'  >> Error during query:t {e}')
        db.rollback()
        return None


def load_labels(db: Session, limit: int, offset: int):
    try:
        return db.query(Label).limit(limit).offset(offset).all()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_labels_by_dataset_notation(db: Session, dataset_id: int, notation_id: int):
    try:
        return db.query(Label).filter(Label.dataset_id == dataset_id).filter(Label.notation_id == notation_id).all()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_notations(db: Session):
    try:
        return db.query(Notation).all()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_trains(db: Session, limit: int, offset: int):
    try:
        return db.query(Train).limit(limit).offset(offset).all()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def count_datasets(db: Session):
    try:
        return db.query(Dataset).count()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return 0


def count_datasets_related_to_user(db: Session, user_id: int):
    try:
        return db.query(Dataset).filter(Dataset.user_id == user_id).count()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return 0


def count_labels(db: Session):
    try:
        return db.query(Label).count()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return 0


def count_trains(db: Session):
    try:
        return db.query(Train).count()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return 0


def load_simple_users(db: Session):
    try:
        return db.query(User).all()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_file_by_dataset_notation(db: Session, dataset_id: int, notation_id: int):
    try:
        return db.query(AudioFile).filter(AudioFile.dataset_id == dataset_id).filter(AudioFile.notation_id == notation_id).all()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def load_file_by_dataset(db: Session, dataset_id: int):
    try:
        return db.query(AudioFile).filter(AudioFile.dataset_id == dataset_id).all()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def delete_file_by_dataset(db: Session, dataset_id: int, file_id: int):
    try:
        db.query(AudioFile).filter(AudioFile.dataset_id == dataset_id).filter(AudioFile.id == file_id).delete()
        db.commit()
    except Exception as e:
        logging.error(f'  >> Error during query: {e}')
        db.rollback()
        return None


def insert_audio_file(db: Session, dataset_id: int, notation_id: int, extension: str) -> AudioFile:
    audio_file = AudioFile(dataset_id=dataset_id, extension=extension, notation_id=notation_id)
    db.add(audio_file)
    db.commit()
    db.refresh(audio_file)
    return audio_file


def update_label(db: Session, file_id: int, label_id: int) -> AudioFile:
    audio_file = db.query(AudioFile).filter(AudioFile.id == file_id).first()
    if audio_file:
        audio_file.label_id = label_id
        db.commit()
        db.refresh(audio_file)
        return {"message": "Label updated successfully", "audio_file_id": audio_file.id, "label_id": label_id}
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")
