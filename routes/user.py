import json
import typing

from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates
from werkzeug.security import generate_password_hash

from database import User
from database.crud import load_users, count_users, load_user_by_username, insert_user, load_user_by_id
from database.database import get_db
from variables import variables

router = APIRouter()
templates = Jinja2Templates(directory=variables.base_dir + "/templates")


@router.get("/", response_class=HTMLResponse)
async def users(
        request: Request,
        page: int = Query(1, alias="page"),
        limit: int = Query(10, alias="limit"),
        db: Session = Depends(get_db)
):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if await is_admin(request):
        offset = (page - 1) * limit

        searched_users = load_users(db, limit, offset, session_user["id"])
        users_count = count_users(db)
        total_pages = 1 if users_count <= limit else (users_count + (limit - 1)) // limit

        # Render the template with the data
        return templates.TemplateResponse(
            'users.html',
            {
                'request': request,
                'users': searched_users or [],
                'total_pages': total_pages,
                'page': page,
                'start_page': max(1, page - 2),
                'end_page': min(total_pages, page + 2),
                'current_user': session_user
            }
        )
    else:
        return RedirectResponse(url="/datasets/", status_code=303)


@router.get("/create", response_class=HTMLResponse)
async def user_create_form(request: Request):
    """
    Handle the creation of a new user.

    GET:
        - Renders the 'user.html' template with a blank user form for admin users.

    POST:
        - Creates a new user based on the form submission.
        - Checks for existing users with the same username or email.
        - If the user exists, flashes a message and re-renders the form.
        - If the user does not exist, inserts the new user and redirects to the 'users' page.

    Returns:
        - A rendered template for GET requests.
        - A redirect to the 'users' page for POST requests.
    """
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if await is_admin(request):
        return templates.TemplateResponse(
            'user.html',
            {
                'request': request,
                'user':
                    User(role_id=2)
                ,
                'current_user': session_user
            }
        )
    else:
        return RedirectResponse(url="/login/", status_code=303)


@router.post("/create", response_class=HTMLResponse)
async def user_create(request: Request, db: Session = Depends(get_db)):
    """
    Handle the creation of a new user.

    GET:
        - Renders the 'user.html' template with a blank user form for admin users.

    POST:
        - Creates a new user based on the form submission.
        - Checks for existing users with the same username or email.
        - If the user exists, flashes a message and re-renders the form.
        - If the user does not exist, inserts the new user and redirects to the 'users' page.

    Returns:
        - A rendered template for GET requests.
        - A redirect to the 'users' page for POST requests.
    """
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if await is_admin(request):
        form = await request.form()

        searched_user = load_user_by_username(
            db,
            form.get("username"),
            form.get("email")
        )

        if searched_user:
            flash(
                request, "User with username {} or email {} already exists".format(
                    form.get("username"), form.get("email")
                )
            )
            new_user = User(role_id=2)
            update_user(form, new_user)
            return templates.TemplateResponse(
                'user.html',
                {
                    'request': request,
                    'user': new_user,
                    'current_user': session_user
                }
            )
        else:
            new_user = User(role_id=2)
            update_user(form, new_user)
            insert_user(db, new_user)
            return RedirectResponse(url="/users/", status_code=303)
    else:
        return RedirectResponse(url="/login/", status_code=303)


@router.get('/profile', response_class=HTMLResponse)
async def user_profile(request: Request, db: Session = Depends(get_db)):
    """
    Handle the login page.

    GET:
        Renders the 'user.html' template to display the profile form.

    POST:
        Processes the profile form submission.
        - If the username or password is missing, flashes an error message and redirects to the login page.
        - Strips leading/trailing whitespace from the username and password.
        - Loads the user by username.
        - Checks the password hash against the provided password.
        - If the credentials are valid, sets the user session and redirects to the dashboard.
        - Otherwise, flashes an error message and re displays the login page.

    Templates:
        - user.html: The template for the profile page.
        """
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    searched_user = load_user_by_id(db, session_user["id"])
    searched_user.password = ""
    return templates.TemplateResponse(
        'user.html',
        {
            'request': request,
            'user': searched_user,
            'is_profile': True,
            'current_user': session_user
        }
    )


@router.get("/{user_id}", response_class=HTMLResponse)
async def user(request: Request, user_id: int, db: Session = Depends(get_db)):
    """
    Handle the user's profile display and update.

    GET:
        - Retrieves the user data for the given user_id.
        - Renders the 'user.html' template with the user data and the current session user.

    POST:
        - Updates the user data based on the form submission.
        - Updates related objects (Tariff, RecognitionConfiguration).
        - Redirects to the 'users' page.

    Args:
        user_id (int): The ID of the user to be retrieved and updated.

    Returns:
        - A rendered template for GET requests.
        - A redirect to the 'users' page for POST requests.
        :param user_id:
        :param request:
    """
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url='/login/', status_code=303)

    if not await is_admin(request):
        return RedirectResponse(url='/profile/', status_code=303)

    searched_user = load_user_by_id(db, user_id)

    searched_user.password = ""
    return templates.TemplateResponse(
        'user.html',
        {
            'request': request,
            'user': searched_user,
            'current_user': session_user
        }
    )


@router.post("/{user_id}", response_class=HTMLResponse)
async def user_update(request: Request, user_id: int, db: Session = Depends(get_db)):
    """
    Handle the user's profile display and update.

    GET:
       - Retrieves the user data for the given user_id.
       - Renders the 'user.html' template with the user data and the current session user.

    POST:
       - Updates the user data based on the form submission.
       - Updates related objects (Tariff, RecognitionConfiguration).
       - Redirects to the 'users' page.

    Args:
       user_id (int): The ID of the user to be retrieved and updated.

    Returns:
       - A rendered template for GET requests.
       - A redirect to the 'users' page for POST requests.
       :param user_id:
       :param request:
    """
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url='/login/', status_code=303)

    searched_user = load_user_by_id(db, user_id)
    if await is_admin(request) or session_user["id"] == searched_user.id:
        form = await request.form()
        update_user(form, searched_user)
        insert_user(db, searched_user)
        if await is_admin(request):
            return RedirectResponse(url='/users/', status_code=303)
        else:
            return RedirectResponse(url='/dashboard/', status_code=303)
    else:
        return RedirectResponse(url='/profile/', status_code=303)


def update_user(form, u: User):
    u.first_name = form.get("first_name", u.first_name)
    u.last_name = form.get("last_name", u.last_name)
    u.email = form.get("email", u.email)
    u.phone = form.get("phone", u.phone)
    u.username = form.get("username", u.username)
    if form.get("password") and form.get("password") != "":
        u.password = generate_password_hash(form.get("password"))


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
