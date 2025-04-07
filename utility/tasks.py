# tasks.py
import asyncio
import datetime
import logging
import os
from asyncio import Queue
from pathlib import Path

import numpy as np
import redis.asyncio as redis  # Використовуємо async redis
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from cnn.cnn import CNNNetwork
from cnn.injector import create_csv
from cnn.voiptime import VoipTimeDataset
from database.core import TrainStatus
from database.crud import load_train_by_id, update_train
from database.database import SessionLocal
from variables import variables

# Налаштування Redis для Pub/Sub
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
redis_pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


async def redis_log_publisher(log_queue: Queue, channel: str, redis_pool):
    """Асинхронно читає логи з черги та публікує в Redis."""
    log = logging.getLogger("redis_log_publisher") # Окремий логер для паблішера
    log.info(f"Starting Redis log publisher for channel {channel}")
    redis_client = None
    try:
        # Створюємо клієнт один раз для ефективності
        redis_client = redis.Redis(connection_pool=redis_pool)
        while True:
            log_entry = None  # Ініціалізація
            try:
                # Очікуємо наступний запис у черзі
                log_entry = await log_queue.get()

                # Перевірка на сигнал завершення
                if log_entry is None:
                    log.info(f"Received stop signal for channel {channel}. Flushing remaining...")
                    log_queue.task_done()
                    break # Вийти з циклу

                # Публікуємо лог
                try:
                    await redis_client.publish(channel, log_entry)
                    # print(f"DEBUG PUBLISHER: Published to {channel}: {log_entry[:50]}") # Debug
                except redis.RedisError as pub_err:
                    log.error(f"Redis error publishing to {channel}: {pub_err}")
                    log_queue.task_done()
                    # Можливо, спробувати перепідключитися або пропустити
                except Exception as e:
                     log.error(f"Unexpected error publishing log to {channel}: {e}", exc_info=True)
                     log_queue.task_done()

                # Повідомляємо черзі, що елемент оброблено
                log_queue.task_done()

            except asyncio.CancelledError:
                log.info(f"Publisher task for {channel} cancelled.")
                if log_entry is not None:  # Якщо ми отримали елемент до скасування
                    log_queue.task_done()
                break
            except Exception as e:
                log.error(f"Error in publisher loop for {channel}: {e}", exc_info=True)
                if log_entry is not None:
                    log_queue.task_done()
                # Додати невелику затримку, щоб уникнути щільного циклу помилок
                await asyncio.sleep(1)
            finally:
                if log_entry is not None:
                    pass

    finally:
        log.info(f"Stopping Redis log publisher for channel {channel}")
        if redis_client:
            try:
                await redis_client.close()
                log.debug(f"Publisher Redis client closed for {channel}")
            except Exception as close_err:
                 log.warning(f"Error closing publisher Redis client for {channel}: {close_err}")


class QueuingRedisHandler(logging.Handler):
    """Кладе відформатовані логи в asyncio.Queue."""
    def __init__(self, log_queue: Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord):
        # Цей метод викликається синхронно
        try:
            log_entry = self.format(record)
            # Кладемо в чергу - це не блокує!
            self.log_queue.put_nowait(log_entry)
            # print(f"DEBUG HANDLER: Put to queue: {log_entry[:50]}") # Debug
        except asyncio.QueueFull:
            # Черга переповнена (малоймовірно з типовим розміром за замовчуванням)
            print(f"WARNING: Log queue is full for channel (task might be too slow or publisher stuck). Log dropped: {self.format(record)}")
        except Exception as e:
            # Інші помилки при форматуванні або put_nowait
            print(f"ERROR: Failed to queue log message: {e}")
            self.handleError(record) # Стандартна обробка помилок логера


def setup_task_logging(log_file_path, redis_channel):
    """Налаштовує логування: у файл та через Queue в Redis."""
    logger = logging.getLogger(f'train_task_{redis_channel}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 1. Логування у файл (залишається як було)
    # ... (ваш код FileHandler) ...
    log_dir = os.path.dirname(log_file_path)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


    # 2. Створення черги та хендлера для Redis
    log_queue = Queue(maxsize=1000) # Обмежимо розмір черги на всякий випадок
    redis_queue_handler = QueuingRedisHandler(log_queue)
    redis_formatter = logging.Formatter('%(asctime)s - %(message)s')
    redis_queue_handler.setFormatter(redis_formatter)
    logger.addHandler(redis_queue_handler)

    # 3. Запуск фонової задачі-паблішера
    publisher_task = asyncio.create_task(
        redis_log_publisher(log_queue, redis_channel, redis_pool),
        name=f"redis_publisher_{redis_channel}" # Даємо ім'я задачі
    )
    logger.info(f"Started background Redis publisher task.")

    # Повертаємо логер та чергу (черга потрібна для сигналу завершення)
    return logger, log_queue, publisher_task


# --- Кінець налаштування логування ---


def train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch, bord_writer, logger):
    epoch_loss = 0.0
    epoch_correct = 0.0
    num_batches = len(data_loader)  # Кількість батчів в епосі

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
    avg_epoch_accuracy = epoch_correct / len(
        data_loader.dataset)  # Або len(data_loader.sampler), якщо використовуєте семплер

    print(
        f"Epoch {epoch + 1} - loss: {avg_epoch_loss:.4f}, accuracy: {avg_epoch_accuracy:.4f}")  # Виводимо середні значення за епоху

    logger.info(f"Epoch {epoch + 1} finished - Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_epoch_accuracy:.4f}")

    # TensorBoard logging for epoch average values
    bord_writer.add_scalar("epoch training loss", avg_epoch_loss, epoch)
    bord_writer.add_scalar("epoch accuracy", avg_epoch_accuracy, epoch)


def train_model(model, data_loader, loss_fn, optimiser, device, epochs, bord_writer, logger):
    logger.info("Starting model training...")
    for i in range(epochs):
        logger.info(f"Starting Epoch {i + 1}/{epochs}")
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device, i, bord_writer, logger)
        print("---------------------------")

    logger.info("Finished training successfully.")
    print("Finished training")


# --- ARQ Задача ---
async def run_training_task(ctx, train_id: int, sample_rate: int, num_samples: int, epochs: int, batch_size: int):
    """Фонова задача для тренування моделі."""
    db = SessionLocal()
    redis_channel = f"train_logs:{train_id}"  # Канал для логів цього тренування
    load_train = load_train_by_id(db, train_id)
    log_queue = None  # Змінна для черги
    publisher_task = None  # Змінна для задачі паблішера
    logger = None
    try:
        if not load_train:
            print(f"[Error] Train with ID {train_id} not found in DB.")  # Логер ще не налаштований
            return

        # --- Налаштування шляхів та логера ---
        train_dir = os.path.join(variables.file_dir, load_train.name)
        log_filename = f"{load_train.name}.log"
        log_filepath = os.path.join(train_dir, "logs", log_filename)
        Path(os.path.dirname(log_filepath)).mkdir(parents=True, exist_ok=True)

        logger, log_queue, publisher_task = setup_task_logging(log_filepath, redis_channel)
        logger.info(f"Starting training task for Train ID: {train_id}, Name: {load_train.name}")
        logger.info(
            f"Parameters: sample_rate={sample_rate}, num_samples={num_samples}, epochs={epochs}, batch_size={batch_size}")

        # Оновлюємо статус у БД
        load_train.status = TrainStatus.RUNNING
        update_train(db, load_train)

        logger.info("Updated train status to RUNNING")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

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

        # --- Запуск тренування ---
        train_model(cnn, train_dataloader, loss_fn, optimiser, device, epochs, bord_writer, logger)

        # --- Збереження моделі ---
        logger.info("Saving trained model...")
        torch_filepath = os.path.join(train_dir, f"{load_train.name}.pth")
        torch.save(cnn.state_dict(), torch_filepath)
        logger.info(f"Model saved to {torch_filepath}")

        bord_writer.close()

        # --- Успішне завершення ---
        load_train.status = TrainStatus.COMPLETED
        update_train(db, load_train)
        logger.info("Training task completed successfully.")

    except Exception as e:
        # --- Обробка помилок ---
        error_message = f"An error occurred during training task for Train ID {train_id}: {e}"
        if logger: logger.error(error_message, exc_info=True)
        else: print(f"[ERROR] ARQ Task: {error_message}"); import traceback; traceback.print_exc()
        load_train.status = TrainStatus.FAILED
        update_train(db, load_train)  # Можна додати сповіщення про помилку
    finally:
        # Завжди закривати сесію БД
        if log_queue:
            logger.info("Signaling log publisher to finish...")
            await log_queue.put(None)
            logger.info("Waiting for log queue to empty...")

            # --- ДОДАТИ ПЕРЕВІРКУ СТАНУ ПАБЛІШЕРА ---
            if publisher_task and not publisher_task.done():
                logger.info(f"Publisher task '{publisher_task.get_name()}' is still running. Waiting for join...")
            elif publisher_task and publisher_task.done():
                logger.warning(f"Publisher task '{publisher_task.get_name()}' already finished before join was called.")
                try:
                    # Перевірити, чи не було помилки
                    exc = publisher_task.exception()
                    if exc:
                        logger.error(f"Publisher task finished with exception: {exc}", exc_info=exc)
                except asyncio.CancelledError:
                    logger.info("Publisher task was cancelled earlier.")
                except asyncio.InvalidStateError:
                    logger.warning("Publisher task state is invalid (could not get exception).")

            # ---------------------------------------------
            try:
                # Додати таймаут до join, щоб уникнути вічного блокування при діагностиці
                await asyncio.wait_for(log_queue.join(), timeout=5.0)  # Таймаут 5 секунд
                logger.info("Log queue empty.")
            except asyncio.TimeoutError:
                logger.error(
                    "TIMEOUT waiting for log queue to empty! Publisher task is likely stuck or task_done() is not called correctly.")
                # Тут можна спробувати отримати стан черги, паблішера тощо для діагностики
                if publisher_task: logger.error(
                    f"Publisher task status at timeout: done={publisher_task.done()}, cancelled={publisher_task.cancelled()}")
                logger.error(f"Queue size at timeout: {log_queue.qsize()}")

            except Exception as q_err:
                logger.error(f"Error signaling/joining log queue: {q_err}")

            # Зупиняємо задачу паблішера, якщо вона ще працює (на випадок помилок)
        if publisher_task and not publisher_task.done():
            publisher_task.cancel()
            try:
                await publisher_task  # Дочекатися завершення після скасування
            except asyncio.CancelledError:
                logger.info("Publisher task successfully cancelled.")
            except Exception as pt_err:
                logger.error(f"Error waiting for cancelled publisher task: {pt_err}")

            # Публікуємо фінальне повідомлення безпосередньо
        try:
            r_pub = redis.Redis(connection_pool=redis_pool)
            await r_pub.publish(redis_channel, "---TASK_FINISHED---")
            await r_pub.close()
            logger.info("Published TASK_FINISHED message.")
        except Exception as pub_err:
            logger.error(f"Error publishing TASK_FINISHED message: {pub_err}")

            # Закриття сесії БД
        if db:
            db.close()
            logger.info("Database session closed.")
