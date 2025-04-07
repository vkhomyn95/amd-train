import asyncio
import datetime
import json
import logging
import os
import shutil
import typing
from pathlib import Path

import numpy as np
import redis.asyncio as redis
import torch
import torchaudio
from arq import ArqRedis
from fastapi import APIRouter, Query, Depends, Form, HTTPException
from sqlalchemy.orm import Session
from sse_starlette import EventSourceResponse
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from starlette.templating import Jinja2Templates
from torch import nn
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from cnn.cnn import CNNNetwork
from cnn.injector import create_csv
from cnn.voiptime import VoipTimeDataset
from database import Train
from database.core import TrainStatus
from database.crud import load_trains, count_trains, load_datasets, insert_train, load_train_by_id, update_train
from database.database import get_db
from utility.arq_worker import REDIS_HOST, REDIS_PORT
from utility.tasks import redis_pool
from variables import variables

router = APIRouter()

templates = Jinja2Templates(directory=variables.base_dir + "/templates")


async def get_arq_redis():
    return await ArqRedis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}")


@router.get("/", response_class=HTMLResponse)
async def trains(
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

        trains = load_trains(db, limit, offset)
        count = count_trains(db)
        total_pages = 1 if count <= limit else (count + (limit - 1)) // limit

        datasets = load_datasets(db, limit, offset)

        return templates.TemplateResponse(
            'trains.html',
            {
                'request': request,
                'trains': trains or [],
                'total_pages': total_pages,
                'page': page,
                'start_page': max(1, page - 2),
                'end_page': min(total_pages, page + 2),
                'datasets': datasets or [],
                'current_user': session_user
            }
        )
    else:
        return RedirectResponse(url="/datasets/", status_code=303)


@router.post("/create", response_class=HTMLResponse)
async def train_create(request: Request, selected_datasets: str = Form(...), db: Session = Depends(get_db)):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    dataset_ids = json.loads(selected_datasets)

    if not dataset_ids:
        flash(request, "Датасет не обраний")
        return RedirectResponse(url="/trains/", status_code=303)

    if await is_admin(request):
        new_train = Train(
            user_id=session_user["id"],
            name="train_" + datetime.datetime.utcnow().strftime("%Y_%m_%d")
        )

        inserted_train = insert_train(db, new_train, dataset_ids)

        return RedirectResponse(url=f"/trains/{inserted_train.id}", status_code=303)
    else:
        return RedirectResponse(url="/login/", status_code=303)


@router.get('/{train_id}', response_class=HTMLResponse)
async def train(request: Request, train_id: int, db: Session = Depends(get_db)):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if not await is_admin(request):
        return RedirectResponse(url="/login/", status_code=303)

    load_train = load_train_by_id(db, train_id)

    if not load_train:
        return RedirectResponse(url="/404", status_code=404)

    return templates.TemplateResponse(
        'train.html',
        {
            'request': request,
            'train': load_train,
            'current_user': session_user,
        }
    )


@router.get('/{train_id}/start', response_class=HTMLResponse)
async def start_train(
        request: Request,
        train_id: int,
        sample_rate: int = Query(800),
        num_samples: int = Query(22050),
        epochs: int = Query(50),
        fold: int = Query(5),
        batch_size: int = Query(128),
        db: Session = Depends(get_db)
):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if not await is_admin(request):
        return RedirectResponse(url="/login/", status_code=303)

    load_train = load_train_by_id(db, train_id)

    if not load_train:
        return RedirectResponse(url="/404", status_code=404)

    load_train.sample_rate = sample_rate
    load_train.num_samples = num_samples
    load_train.epochs = epochs
    load_train.batch_size = batch_size
    load_train.fold = fold
    update_train(db, load_train)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("===Train: %s Using device: %s for train", load_train.name, device)

    try:
        target_dir = Path(variables.file_dir) / load_train.name

        if target_dir.is_dir():
            shutil.rmtree(target_dir)
    except Exception as e:
        pass

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=load_train.sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    create_csv(db, load_train, False)

    csv_filename = f"dataset_{datetime.datetime.utcnow().strftime('%Y_%m_%d')}.csv"
    csv_filepath = os.path.join(variables.file_dir, load_train.name, csv_filename)

    Path(os.path.join(variables.file_dir, load_train.name, "audio")).mkdir(parents=True, exist_ok=True)

    usd = VoipTimeDataset(
        csv_filepath,
        os.path.join(variables.file_dir, load_train.name, "audio"),
        mel_spectrogram,
        load_train.sample_rate,
        load_train.num_samples,
        device
    )

    # Отримуємо мітки, які реально використовуються моделлю (з колонки 'target')
    labels_target = usd.annotations["target"]  # Або usd.get_labels(), якщо ви зміните метод get_labels()

    labels_unique, counts = np.unique(labels_target, return_counts=True)
    # Тепер labels_unique буде [0, 1]

    logging.info("===Train: {} Unique target labels: {}".format(load_train.name, labels_unique))

    # Розраховуємо ваги для класів 0 та 1
    # Перевірка, щоб уникнути ділення на нуль, якщо якийсь клас відсутній (хоча з WeightedRandomSampler це малоймовірно)
    total_samples = len(labels_target)
    class_weights_values = [total_samples / count if count > 0 else 0 for count in counts]

    # Створюємо словник: мітка_класу -> вага
    class_weights_dict = {label: weight for label, weight in zip(labels_unique, class_weights_values)}

    logging.info("===Train: {} Class weights dict: {}".format(load_train.name, class_weights_dict))

    # Призначаємо кожному прикладу відповідну вагу на основі його мітки 'target'
    example_weights = [class_weights_dict[label] for label in labels_target]

    # Передаємо ваги для кожного прикладу в семплер
    sampler = WeightedRandomSampler(example_weights, len(labels_target))

    train_dataloader = DataLoader(usd, batch_size=load_train.batch_size, sampler=sampler)

    cnn = CNNNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(cnn.parameters(), lr=0.000001)

    tensor_board_log = os.path.join(variables.file_dir, load_train.name, "tensorboard")

    Path(tensor_board_log).mkdir(parents=True, exist_ok=True)

    bord_writer = SummaryWriter(log_dir=tensor_board_log)

    train_model(cnn, train_dataloader, loss_fn, optimiser, device, load_train.epochs, bord_writer)

    torch_filepath = os.path.join(variables.file_dir, load_train.name, f"{load_train.name}.pth")

    torch.save(cnn.state_dict(), torch_filepath)

    bord_writer.close()

    return templates.TemplateResponse(
        'train.html',
        {
            'request': request,
            'train': load_train,
            'current_user': session_user,
        }
    )


def train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch, bord_writer):
    epoch_loss = 0.0
    epoch_correct = 0.0
    num_batches = len(data_loader) # Кількість батчів в епосі

    for i, (input, target) in enumerate(data_loader):
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Accumulate loss and correct predictions
        epoch_loss += loss.item()
        _, predicted = torch.max(prediction.data, 1)
        epoch_correct += (predicted == target).sum().item()

    # Calculate average loss and accuracy for the epoch
    avg_epoch_loss = epoch_loss / num_batches
    avg_epoch_accuracy = epoch_correct / len(data_loader.dataset) # Або len(data_loader.sampler), якщо використовуєте семплер

    print(f"Epoch {epoch+1} - loss: {avg_epoch_loss:.4f}, accuracy: {avg_epoch_accuracy:.4f}") # Виводимо середні значення за епоху

    # TensorBoard logging for epoch average values
    bord_writer.add_scalar("epoch training loss", avg_epoch_loss, epoch)
    bord_writer.add_scalar("epoch accuracy", avg_epoch_accuracy, epoch)


def train_model(model, data_loader, loss_fn, optimiser, device, epochs, bord_writer):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device, i, bord_writer)
        print("---------------------------")
    print("Finished training")


@router.post('/{train_id}/start_background')
async def start_train_background(
        request: Request,
        train_id: int,
        sample_rate: int = Query(800),
        num_samples: int = Query(22050),
        epochs: int = Query(50),
        fold: int = Query(5),
        batch_size: int = Query(128),
        db: Session = Depends(get_db),
        arq_redis: ArqRedis = Depends(get_arq_redis) # Ін'єкція ARQ клієнта
):
    session_user = await get_user(request) # Ваша функція автентифікації
    if not session_user or not await is_admin(request): # Ваша перевірка адміна
        # Замість редіректу, краще повернути помилку API для JS
        raise HTTPException(status_code=403, detail="Authentication or permission failed")

    load_train = load_train_by_id(db, train_id)
    if not load_train:
        raise HTTPException(status_code=404, detail="Train not found")

    # Перевірка, чи тренування вже не запущене
    if load_train.status == TrainStatus.RUNNING or load_train.status == TrainStatus.QUEUED:
         raise HTTPException(status_code=400, detail="Training is already running or queued")

    try:
        target_dir = Path(variables.file_dir) / load_train.name

        if target_dir.is_dir():
            shutil.rmtree(target_dir)
    except Exception as e:
        pass

    # Оновлюємо параметри в БД (опціонально, якщо потрібно їх зберігати перед запуском)
    load_train.sample_rate = sample_rate
    load_train.num_samples = num_samples
    load_train.epochs = epochs
    load_train.batch_size = batch_size
    load_train.fold = fold
    update_train(db, load_train)

    # Додаємо задачу в чергу ARQ
    job = await arq_redis.enqueue_job(
        'run_training_task', # Назва функції-задачі в ARQ
        train_id=train_id,   # Аргументи для функції
        sample_rate=sample_rate,
        num_samples=num_samples,
        epochs=epochs,
        batch_size=batch_size,
        _job_id=f"train_{train_id}_{datetime.datetime.utcnow().timestamp()}" # Унікальний ID задачі
    )

    # Оновлюємо статус у БД на "В черзі" та зберігаємо Job ID
    load_train.status = TrainStatus.QUEUED
    load_train.job_id = job.job_id
    update_train(db, load_train)

    logging.info(f"Enqueued training task for Train ID: {train_id} with Job ID: {job.job_id}")

    # Повертаємо успішну відповідь клієнту (JS)
    # Можна повернути ID задачі або просто успіх
    return {"message": "Training started successfully", "train_id": train_id, "job_id": job.job_id}


@router.get('/{train_id}/logs/stream')
async def stream_logs(request: Request, train_id: int):
    """Ендпоінт для потокової передачі логів через Server-Sent Events з надійною обробкою помилок."""

    session_user = await get_user(request)
    if not session_user:
        raise HTTPException(status_code=403, detail="Not authenticated")

    redis_channel = f"train_logs:{train_id}"

    async def event_generator():
        r = None
        pubsub = None
        logging.info(f"SSE stream requested for channel: {redis_channel}")

        try:
            # --- Блок 1: Спроба підключення та підписки ---
            try:
                r = redis.Redis(connection_pool=redis_pool)
                # Перевірка з'єднання перед підпискою (добра практика)
                await r.ping()
                logging.info(f"Redis connection successful for {redis_channel}")
                pubsub = r.pubsub()
                await pubsub.subscribe(redis_channel)
                logging.info(f"Subscribed to Redis channel: {redis_channel}")
            except redis.RedisError as conn_err:
                logging.error(f"Failed to connect/subscribe to Redis channel {redis_channel}: {conn_err}", exc_info=True)
                yield {"event": "error", "data": f"Log source connection error: {conn_err}"}
                return # Зупинити генератор
            except Exception as setup_err: # Інші помилки налаштування
                 logging.error(f"Unexpected error during SSE setup for {redis_channel}: {setup_err}", exc_info=True)
                 yield {"event": "error", "data": f"Error setting up log stream: {setup_err}"}
                 return

            # --- Блок 2: Відправка стартового повідомлення ---
            yield {"event": "message", "data": "--- Listening for logs... ---"}

            # --- Блок 3: Основний цикл обробки повідомлень ---
            while True:
                # 3a: Перевірка відключення клієнта
                if await request.is_disconnected():
                    logging.info(f"SSE client disconnected for channel: {redis_channel}")
                    break # Нормальний вихід

                message = None
                try:
                    # 3b: Отримання повідомлення з Redis
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
                except redis.RedisError as redis_err:
                     logging.error(f"Redis error while getting message on {redis_channel}: {redis_err}", exc_info=True)
                     yield {"event": "error", "data": f"Log source connection error: {redis_err}"}
                     await asyncio.sleep(2) # Затримка перед виходом з циклу
                     break # Вийти з циклу, щоб викликати finally
                except Exception as loop_err: # Інші помилки в циклі (до yield)
                     logging.error(f"Unexpected error in SSE loop for {redis_channel}: {loop_err}", exc_info=True)
                     yield {"event": "error", "data": f"Internal error in log stream: {loop_err}"}
                     await asyncio.sleep(2)
                     break # Вийти з циклу, щоб викликати finally

                # 3c: Обробка отриманого повідомлення
                if message and message.get("type") == "message":
                    log_data = message.get('data', '') # Безпечне отримання

                    # Конвертація в рядок (якщо потрібно)
                    if isinstance(log_data, bytes):
                        log_data = log_data.decode('utf-8', errors='replace')
                    elif not isinstance(log_data, str):
                        log_data = str(log_data)

                    # Перевірка на сигнал завершення
                    if log_data == "---TASK_FINISHED---":
                        logging.info(f"Received TASK_FINISHED signal for channel: {redis_channel}")
                        yield {"event": "message", "data": "--- Training task finished or stopped. ---"}
                        break # Нормальний вихід

                    # 3d: Відправка логу клієнту
                    try:
                        yield {"event": "message", "data": log_data}
                    except Exception as yield_err:
                         # Ймовірно, клієнт відключився під час yield
                         logging.warning(f"Error yielding data on {redis_channel} (client likely disconnected): {yield_err}")
                         break # Вийти з циклу

                # 3e: Невелика пауза, якщо не було повідомлення
                await asyncio.sleep(0.1) # Зменшив паузу

        # --- Блок 4: Обробка винятків для всього генератора ---
        except asyncio.CancelledError:
            # Клієнт відключився або сервер зупиняється
            logging.info(f"SSE task cancelled for channel: {redis_channel}")
        except Exception as e:
            # Загальний перехоплювач на випадок, якщо щось пропустили вище
            logging.error(f"Unhandled exception in SSE generator for channel {redis_channel}: {e}", exc_info=True)
            try:
                 yield {"event": "error", "data": f"An critical error occurred in the log stream: {e}"}
            except Exception: pass # Ігнорувати, якщо yield не вдасться

        # --- Блок 5: Гарантоване очищення ресурсів ---
        finally:
            logging.info(f"Cleaning up SSE resources for channel: {redis_channel}")
            if pubsub:
                try:
                    await pubsub.unsubscribe(redis_channel)
                    logging.debug(f"Unsubscribed from {redis_channel}")
                except Exception as unsub_e:
                    logging.warning(f"Error unsubscribing from Redis channel {redis_channel}: {unsub_e}")
            if r:
                try:
                    await r.close()
                    logging.debug(f"Closed Redis connection for {redis_channel}")
                except Exception as close_e:
                    logging.warning(f"Error closing Redis connection for channel {redis_channel}: {close_e}")
            logging.info(f"Finished SSE stream cleanup for channel: {redis_channel}")

    return EventSourceResponse(event_generator())


@router.get('/{train_id}/logs/file', response_class=PlainTextResponse)
async def get_logs_from_file(
    request: Request,
    train_id: int,
    db: Session = Depends(get_db)
):
    session_user = await get_user(request)
    if not session_user:
        raise HTTPException(status_code=403, detail="Not authenticated")

    logging.info(f"Request received to fetch log file for train_id: {train_id}")

    load_train = load_train_by_id(db, train_id)
    if not load_train:
        logging.warning(f"Train not found for train_id: {train_id}")
        raise HTTPException(status_code=404, detail=f"Train with id {train_id} not found")

    log_file_path_str = os.path.join(variables.file_dir, load_train.name, "logs", f"{load_train.name}.log")
    if not log_file_path_str:
        logging.warning(f"Log file path not found in DB for train_id: {train_id}")
        raise HTTPException(status_code=404, detail=f"Log file path not recorded for train {train_id}")

    log_file_path = Path(log_file_path_str)
    logging.info(f"Attempting to read log file: {log_file_path}")

    if not log_file_path.is_file():
        logging.error(f"Log file does not exist or is not a file: {log_file_path}")
        raise HTTPException(status_code=404, detail=f"Log file not found on server for train {train_id}")

    try:
        # Читаємо вміст файлу
        log_content = log_file_path.read_text(encoding='utf-8')
        logging.info(f"Successfully read {len(log_content)} bytes from {log_file_path}")
        # Повертаємо вміст як простий текст
        return PlainTextResponse(content=log_content)
    except OSError as e:
        logging.error(f"OS error reading log file {log_file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading log file: {e}")
    except Exception as e:
        logging.error(f"Unexpected error reading log file {log_file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while reading the log file.")


@router.get('/{train_id}/download_csv')
async def download_csv(
        request: Request,
        train_id: int,
        sample_rate: int = Query(8000),
        num_samples: int = Query(22050),
        epochs: int = Query(50),
        fold: int = Query(5),
        batch_size: int = Query(128),
        db: Session = Depends(get_db)
):
    session_user = await get_user(request)

    if not session_user:
        return RedirectResponse(url="/login/", status_code=303)

    if not await is_admin(request):
        return RedirectResponse(url="/login/", status_code=303)

    load_train = load_train_by_id(db, train_id)

    if not load_train:
        return RedirectResponse(url="/404", status_code=404)

    load_train.sample_rate = sample_rate
    load_train.num_samples = num_samples
    load_train.epochs = epochs
    load_train.batch_size = batch_size
    load_train.fold = fold
    update_train(db, load_train)

    return create_csv(db, load_train)


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