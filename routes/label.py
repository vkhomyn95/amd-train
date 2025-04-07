import json
import typing

from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates

from database.core import Label
from database.crud import count_datasets, load_labels, load_notations, load_datasets_all, insert_label, \
    load_label_by_id, load_datasets_all_related_to_user
from database.database import get_db
from variables import variables

router = APIRouter()

templates = Jinja2Templates(directory=variables.base_dir + "/templates")


@router.get("/", response_class=HTMLResponse)
async def labels(
        request: Request,
        page: int = Query(1, alias="page"),
        limit: int = Query(10, alias="limit"),
        db: Session = Depends(get_db)
):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    offset = (page - 1) * limit

    labels = load_labels(db, limit, offset)
    count = count_datasets(db)
    total_pages = 1 if count <= limit else (count + (limit - 1)) // limit

    notations = load_notations(db)

    datasets = load_datasets_all(db) if is_admin(request) else load_datasets_all_related_to_user(db, session_user["id"])

    return templates.TemplateResponse(
        'labels.html',
        {
            'request': request,
            'notations': notations or [],
            'total_pages': total_pages,
            'page': page,
            'start_page': max(1, page - 2),
            'end_page': min(total_pages, page + 2),
            'labels': labels,
            'datasets': datasets or [],
            'current_user': session_user
        }
    )


@router.post("/create", response_class=HTMLResponse)
async def label_create(request: Request, db: Session = Depends(get_db)):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if await is_admin(request):
        form = await request.form()

        label_ent = Label(name=form.get('name'), description=form.get('description'), dataset_id=form.get('dataset_id'), notation_id=form.get('notation_id'), user_id=session_user["id"])

        inserted = insert_label(db, label_ent)
        return RedirectResponse(url=f"/labels/{inserted.id}", status_code=303)
    else:
        return RedirectResponse(url="/login/", status_code=303)


@router.post("/update", response_class=HTMLResponse)
async def label_update(request: Request, db: Session = Depends(get_db)):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if await is_admin(request):
        form = await request.form()

        load_label = load_label_by_id(db, int(form.get("label_id")))

        if not load_label:
            return RedirectResponse(url="/404", status_code=404)

        load_label.name = form.get('name', load_label.name)
        load_label.description = form.get('description', load_label.description)
        load_label.dataset_id = form.get('dataset_id', load_label.dataset_id)
        load_label.notation_id = form.get('notation_id', load_label.notation_id)

        insert_label(db, load_label)
        return RedirectResponse(url=f"/labels", status_code=303)
    else:
        return RedirectResponse(url="/login/", status_code=303)


@router.get('/{label_id}', response_class=HTMLResponse)
async def label(request: Request, label_id: int, db: Session = Depends(get_db)):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if not await is_admin(request):
        return RedirectResponse(url="/login/", status_code=303)

    load_label = load_label_by_id(db, label_id)

    if not load_label:
        return RedirectResponse(url="/404", status_code=404)

    notations = load_notations(db)

    datasets = load_datasets_all(db)

    return templates.TemplateResponse(
        'label.html',
        {
            'request': request,
            'notations': notations or [],
            'datasets': datasets or [],
            'label': load_label,
            'current_user': session_user
        }
    )


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


def flash(request: Request, message: typing.Any, category: str = "primary") -> None:
    if "_messages" not in request.session:
        request.session["_messages"] = []
        request.session["_messages"].append({"message": message, "category": category})


def get_flashed_messages(request: Request):
    return request.session.pop("_messages") if "_messages" in request.session else []


templates.env.globals["get_flashed_messages"] = get_flashed_messages