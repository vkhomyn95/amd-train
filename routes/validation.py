import json
import shutil
import tempfile
import typing
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates

from cnn.class_mapping import get_class_mapping_for_train
from cnn.inference import ModelInference
from database.core import TrainStatus
from database.crud import load_trains_all, load_train_by_id
from database.database import get_db
from variables import variables

router = APIRouter()

templates = Jinja2Templates(directory=variables.base_dir + "/templates")


@router.get("/", response_class=HTMLResponse)
async def validation(
        request: Request,
        db: Session = Depends(get_db)
):
    """Відображає сторінку валідації з формою вибору моделі."""
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    # Завантажуємо тільки успішно навчені моделі
    all_trains = load_trains_all(db)
    trains = [train for train in all_trains if train.status == 'TrainStatus.COMPLETED']

    return templates.TemplateResponse(
        'validation.html',
        {
            'request': request,
            'trains': trains or [],
            'results': [],
            'current_user': session_user
        }
    )


@router.post("/check", response_class=HTMLResponse)
async def check_model(
        request: Request,
        train_id: int = Form(...),
        files: typing.List[UploadFile] = File(...),
        db: Session = Depends(get_db)
):
    """
    Виконує валідацію навченої моделі на завантажених аудіофайлах.
    """
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    # 1. Завантажуємо інформацію про тренування
    train_info = load_train_by_id(db, train_id)
    if not train_info:
        flash(request, "Модель не знайдена", "danger")
        return RedirectResponse(url="/validation", status_code=303)

    # Перевіряємо статус моделі
    if train_info.status != 'TrainStatus.COMPLETED':
        flash(request, "Модель ще не завершила навчання або завершилася з помилкою", "danger")
        return RedirectResponse(url="/validation", status_code=303)

    # Шлях до файлу моделі
    model_path = Path(variables.file_dir) / train_info.name / f"{train_info.name}.pth"

    if not model_path.exists():
        flash(request, f"Файл моделі не знайдено: {model_path}", "danger")
        return RedirectResponse(url="/validation", status_code=303)

    results = []

    # Створюємо тимчасову директорію для збереження завантажених файлів
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Завантажуємо маппінг класів для цього тренування
            try:
                class_mapping = get_class_mapping_for_train(train_info.name, variables.file_dir)
                logger_msg = f"Loaded class mapping: {class_mapping}"
            except Exception as e:
                # Якщо маппінг не знайдено, використовуємо дефолтний для бінарної класифікації
                class_mapping = {
                    1: "human",
                    0: "voicemail"
                }
                logger_msg = f"Using default class mapping (error loading saved mapping: {e})"

            # Параметри повинні співпадати з тими, що використовувалися при тренуванні
            sample_rate = train_info.sample_rate  # Може бути збережено в train_info
            num_samples = train_info.num_samples  # Може бути збережено в train_info

            # Ініціалізуємо інференс
            inference = ModelInference(
                model_path=str(model_path),
                class_mapping=class_mapping,
                sample_rate=sample_rate,
                num_samples=num_samples,
                device=None  # або "cuda" якщо є GPU
            )

            # Обробляємо кожен файл
            for idx, file in enumerate(files):
                if not file.filename:
                    continue

                temp_file_path = Path(temp_dir) / file.filename

                # Зберігаємо файл на диск
                with open(temp_file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                # Проводимо передбачення
                try:
                    class_idx, predicted_label, confidence = inference.predict(str(temp_file_path))

                    results.append({
                        "index": idx + 1,
                        "filename": file.filename,
                        "class_index": class_idx,
                        "label": predicted_label,
                        "confidence": f"{confidence:.2%}",
                        "error": None
                    })
                except Exception as e:
                    results.append({
                        "index": idx + 1,
                        "filename": file.filename,
                        "class_index": None,
                        "label": None,
                        "confidence": None,
                        "error": str(e)
                    })

            # Якщо все пройшло успішно
            if results:
                # Підрахунок статистики
                successful = [r for r in results if r["error"] is None]
                if successful:
                    avg_confidence = sum(float(r["confidence"].rstrip('%')) for r in successful) / len(successful)
                    flash(request,
                          f"Оброблено {len(successful)} з {len(results)} файлів. "
                          f"Середня впевненість: {avg_confidence:.2f}%",
                          "success")
                else:
                    flash(request, "Всі файли оброблені з помилками", "warning")

        except FileNotFoundError as e:
            flash(request, f"Файл моделі не знайдено: {e}", "danger")
            results = []
        except Exception as e:
            flash(request, f"Помилка під час інференсу: {e}", "danger")
            results = []

    # Повертаємо ту саму сторінку, але з результатами
    all_trains = load_trains_all(db)
    trains = [train for train in all_trains if train.status == 'TrainStatus.COMPLETED']

    return templates.TemplateResponse(
        'validation.html',
        {
            'request': request,
            'trains': trains or [],
            'results': results,
            'selected_train_id': train_id,
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
    user_json = request.session.get("user")
    if user_json:
        return json.loads(user_json)
    return None


async def is_admin(request: Request):
    """
    Check if the current session user is an admin.

    Args:
        request (Request): The current request object.

    Returns:
        bool: True if the user is an admin, else False.
    """
    user_data = await get_user(request)
    if user_data and "role" in user_data:
        return user_data["role"]["name"] == 'admin'
    return False


def flash(request: Request, message: typing.Any, category: str = "primary") -> None:
    """Додає flash-повідомлення в сесію."""
    if "_messages" not in request.session:
        request.session["_messages"] = []
    request.session["_messages"].append({"message": message, "category": category})


def get_flashed_messages(request: Request):
    """Отримує та видаляє flash-повідомлення з сесії."""
    return request.session.pop("_messages") if "_messages" in request.session else []


templates.env.globals["get_flashed_messages"] = get_flashed_messages
