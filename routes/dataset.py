import json
import os
from pathlib import Path
from typing import List

import pycountry
from fastapi import APIRouter, Query, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, FileResponse
from starlette.templating import Jinja2Templates

from database import Dataset
from database.crud import count_datasets, load_datasets, load_dataset_by_country, insert_dataset, load_dataset_by_id, \
    insert_audio_file, load_file_by_dataset_notation, load_labels_by_dataset_notation, update_label, \
    load_datasets_related_to_user, count_datasets_related_to_user, load_dataset_by_country_user_id, \
    load_dataset_by_id_related_to_user, delete_file_by_dataset
from database.database import get_db
from routes.auth import flash
from variables import variables

router = APIRouter()

templates = Jinja2Templates(directory=variables.base_dir + "/templates")


@router.get("/", response_class=HTMLResponse)
async def datasets(
        request: Request,
        page: int = Query(1, alias="page"),
        limit: int = Query(10, alias="limit"),
        db: Session = Depends(get_db)
):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    offset = (page - 1) * limit
    if await is_admin(request):
        datasets = load_datasets(db, limit, offset)
        count = count_datasets(db)
    else:
        datasets = load_datasets_related_to_user(db, session_user["id"], limit, offset)
        count = count_datasets_related_to_user(db, session_user["id"])

    total_pages = 1 if count <= limit else (count + (limit - 1)) // limit
    return templates.TemplateResponse(
        'datasets.html',
        {
            'request': request,
            'datasets': datasets or [],
            'total_pages': total_pages,
            'page': page,
            'start_page': max(1, page - 2),
            'end_page': min(total_pages, page + 2),
            "countries": list(pycountry.countries),
            'current_user': session_user
        }
    )


@router.post("/create", response_class=HTMLResponse)
async def dataset_create(request: Request, db: Session = Depends(get_db)):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    form = await request.form()

    searched_dataset = load_dataset_by_country_user_id(db, form.get("country"), session_user["id"])

    if searched_dataset:
        flash(request, "Dataset {} already exists".format(form.get("country")))
        return RedirectResponse(url=f"/datasets/{searched_dataset.id}", status_code=303)
    else:
        new_dataset = Dataset(user_id=session_user["id"], country=form.get("country"))
        searched_dataset = insert_dataset(db, new_dataset)
        return RedirectResponse(url=f"/datasets/{searched_dataset.id}", status_code=303)


@router.get('/{dataset_id}', response_class=HTMLResponse)
async def dataset(request: Request, dataset_id: int, directory_folder: str = Query(None), db: Session = Depends(get_db)):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if await is_admin(request):
        load_dataset = load_dataset_by_id(db, dataset_id)
    else:
        load_dataset = load_dataset_by_id_related_to_user(db, dataset_id, session_user["id"])

    if not load_dataset:
        return RedirectResponse(url="/404", status_code=404)

    root_directory = os.path.join(variables.file_dir, str(load_dataset.user_id), load_dataset.country)

    Path(root_directory).mkdir(parents=True, exist_ok=True)

    subdirectories = ["human", "voicemail", "ring", "audio"]
    for subdir in subdirectories:
        Path(os.path.join(root_directory, subdir)).mkdir(parents=True, exist_ok=True)

    current_directory = root_directory
    if directory_folder:
        current_directory = os.path.join(root_directory, directory_folder)

    if directory_folder == "human":
        notation_id = 1
    elif directory_folder == "voicemail":
        notation_id = 2
    elif directory_folder == "ring":
        notation_id = 3
    else:
        notation_id = None

    db_files = load_file_by_dataset_notation(db, load_dataset.id, notation_id)
    db_filenames = {"{}{}".format(v.id, v.extension): v for v in db_files}

    items = get_files_in_directory(current_directory)
    files = []
    directories = []

    for item in items:
        item_path = os.path.join(current_directory, item)
        if os.path.isfile(item_path):
            is_stored = db_filenames.get(item, None)
            if is_stored:
                files.append({"filename": item, "stored": is_stored, "label_id": is_stored.label_id, "id": is_stored.id})
            else:
                files.append({"filename": item, "stored": False, "label_id": 0, "id": None})
        elif os.path.isdir(item_path):
            directories.append(item)

    labels = []

    if len(files) > 0:
        labels = load_labels_by_dataset_notation(db, dataset_id, notation_id)

    return templates.TemplateResponse(
        'dataset.html',
        {
            'request': request,
            'dataset': load_dataset,
            'labels': labels,
            'current_user': session_user,
            'files': files,
            'directories': directories,
            'current_directory': current_directory,
            'base_directory': root_directory
        }
    )


@router.post('/{dataset_id}/upload')
async def upload_file(dataset_id: int, request: Request, directory_folder: str = Form(""), files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if not await is_admin(request):
        return RedirectResponse(url="/login/", status_code=303)

    load_dataset = load_dataset_by_id(db, dataset_id)

    if not load_dataset:
        return RedirectResponse(url="/404", status_code=404)

    root_directory = os.path.join(variables.file_dir, str(load_dataset.user_id), load_dataset.country)

    current_directory = root_directory
    if directory_folder:
        current_directory = os.path.join(root_directory, 'audio' if directory_folder == root_directory else directory_folder)

    if directory_folder == "human":
        notation_id = 1
    elif directory_folder == "voicemail":
        notation_id = 2
    elif directory_folder == "ring":
        notation_id = 3
    else:
        notation_id = None

    for file in files:
        try:
            if file.size > 0:
                file_extension = os.path.splitext(file.filename)[1]
                audio_file = insert_audio_file(db, dataset_id=dataset_id, notation_id=notation_id, extension=file_extension)
                new_filename = str(audio_file.id) + file_extension
                file_path = os.path.join(current_directory, new_filename)

                contents = await file.read()
                with open(file_path, 'wb') as f:
                    f.write(contents)
        except Exception as e:
            print(e)
            return {"message": f"There was an error uploading the file {file.filename}"}
        finally:
            await file.close()

    return RedirectResponse(url=f"/datasets/{dataset_id}?directory_folder={directory_folder}", status_code=303)


@router.post('/{dataset_id}/audio_files/{file_id}/update_label')
async def update_audio_file_label(request: Request, dataset_id: int, file_id: int, label_id: int = Form(...), db: Session = Depends(get_db)):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if not await is_admin(request):
        return RedirectResponse(url="/login/", status_code=303)

    try:
        return update_label(db, file_id, label_id)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/{dataset_id}/download/{filename}')
async def download_file(request: Request, dataset_id: int, filename: str, directory_folder: str = Query(None), db: Session = Depends(get_db)):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if not await is_admin(request):
        return RedirectResponse(url="/login/", status_code=303)

    load_dataset = load_dataset_by_id(db, dataset_id)

    if not load_dataset:
        return RedirectResponse(url="/404", status_code=404)

    root_directory = os.path.join(variables.file_dir, str(load_dataset.user_id), load_dataset.country)

    current_directory = root_directory
    if directory_folder:
        current_directory = os.path.join(root_directory, directory_folder)

    file_path = os.path.join(current_directory, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename, media_type='application/octet-stream')
    else:
        return {"message": "File not found"}


@router.get('/{dataset_id}/play/{filename}')
async def play_file(request: Request, dataset_id: int, filename: str, directory_folder: str = Query(None), db: Session = Depends(get_db)):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if not await is_admin(request):
        return RedirectResponse(url="/login/", status_code=303)

    load_dataset = load_dataset_by_id(db, dataset_id)

    if not load_dataset:
        return RedirectResponse(url="/404", status_code=404)

    root_directory = os.path.join(variables.file_dir, str(load_dataset.user_id), load_dataset.country)

    current_directory = root_directory
    if directory_folder:
        current_directory = os.path.join(root_directory, directory_folder)

    file_path = os.path.join(current_directory, filename)
    return {"file_path": f"/datasets/{dataset_id}/download/{filename}?directory_folder={directory_folder}"}


@router.get('/{dataset_id}/delete/{filename}')
async def delete_file(request: Request, dataset_id: int, filename: str, directory_folder: str = Query(None), db: Session = Depends(get_db)):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if not await is_admin(request):
        flash(request, "Permission Denied")
        return

    load_dataset = load_dataset_by_id(db, dataset_id)

    if not load_dataset:
        return RedirectResponse(url="/404", status_code=404)

    root_directory = os.path.join(variables.file_dir, str(load_dataset.user_id), load_dataset.country)

    current_directory = root_directory
    if directory_folder:
        current_directory = os.path.join(root_directory, directory_folder)

    file_path = os.path.join(current_directory, filename)
    try:
        os.remove(file_path)
    except Exception:
        return {"message": "There was an error deleting the file"}

    parts = filename.rsplit('.', 1)

    file_id = parts[0]

    delete_file_by_dataset(db, dataset_id, int(file_id))

    return RedirectResponse(url=f"/datasets/{dataset_id}?directory_folder={directory_folder}", status_code=303)


def get_files_in_directory(directory: str):
    try:
        items = os.listdir(directory)
        return items
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error reading directory {directory}: {e}")
        return []


def get_save_directory(received_date):
    # Create directory path based on received date (year/month/day)
    year, month, day = received_date.split('T')[0].split("-")
    save_dir = os.path.join(variables.file_dir, year, month, day)

    # Create the directories if they do not exist
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


async def get_user(request: Request) -> dict:
    """
    Retrieve the current session user.

    Args:
        request (Request): The current request object.

    Returns:
        dict: The session user if exists, else None.
    """
    return json.loads(request.session.get("user"))


async def is_admin(request: Request):
    """
    Check if the current session user is an admin.

    Args:
        request (Request): The current request object.

    Returns:
        bool: True if the user is an admin, else False.
    """
    user_data = await get_user(request)
    return user_data["role"]["name"] == 'admin'
