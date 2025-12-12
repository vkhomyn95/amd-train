import asyncio
import datetime
import logging
import os
import shutil
from pathlib import Path
from queue import Queue, Empty  # ⭐ Звичайна синхронна Queue

import numpy as np
import pandas as pd
import redis.asyncio as redis
import torch
import torchaudio
from torch import nn
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from cnn.class_mapping import save_class_mapping
from cnn.cnn import CNNNetwork
from cnn.injector import create_csv
from cnn.voiptime import VoipTimeDataset
from database.core import TrainStatus
from database.database import SessionLocal
from database.crud import load_train_by_id, update_train
from variables import variables

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
redis_pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


class RedisLogHandler(logging.Handler):
    """Handler для публікації логів в Redis через синхронну чергу"""

    def __init__(self, log_queue: Queue):  # ⭐ Змінено на Queue
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            log_entry = self.format(record)
            # ⭐ Використовуємо синхронну put (без _nowait)
            self.log_queue.put(log_entry, block=False)
            print(f"[HANDLER] Queued log: {log_entry[:80]}...")
        except Exception as e:
            print(f"[HANDLER] Error queueing log: {e}")


async def redis_log_publisher(redis_channel: str, log_queue: Queue, stop_event: asyncio.Event):
    """Публікує логи з синхронної черги в Redis в реальному часі"""

    r = None
    try:
        r = redis.Redis(connection_pool=redis_pool)
        await r.ping()
        print(f"[REDIS PUBLISHER] Connected to Redis for channel {redis_channel}")

        while not stop_event.is_set() or not log_queue.empty():
            try:
                # ⭐ Використовуємо синхронний get з timeout
                log_message = log_queue.get(timeout=0.5)

                # Публікуємо НЕГАЙНО
                subscribers = await r.publish(redis_channel, log_message)
                print(f"[REDIS PUBLISHER] Published: {log_message[:80]}... ({subscribers} subscribers)")

                log_queue.task_done()

            except Empty:
                # Timeout - перевіряємо stop_event і продовжуємо
                await asyncio.sleep(0.1)
                continue
            except Exception as e:
                print(f"[REDIS PUBLISHER] Error publishing log: {e}")
                await asyncio.sleep(0.1)

        # Публікуємо сигнал завершення
        await r.publish(redis_channel, "---TASK_FINISHED---")
        print(f"[REDIS PUBLISHER] Sent TASK_FINISHED signal to {redis_channel}")

    except Exception as e:
        print(f"[REDIS PUBLISHER] Fatal error: {e}")
    finally:
        if r:
            await r.close()
        print(f"[REDIS PUBLISHER] Closed connection for {redis_channel}")


def train_single_epoch(model, data_loader, loss_fn, optimiser, device, epoch, logger):
    """Тренування однієї епохи з логуванням"""
    epoch_loss = 0.0
    epoch_correct = 0
    num_batches = len(data_loader)

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

        # Логуємо прогрес кожні 10 батчів
        if i % 10 == 0:
            logger.info(f"Epoch {epoch + 1} - Batch {i}/{num_batches} - Current loss: {loss.item():.4f}")

    # Calculate average loss and accuracy for the epoch
    avg_epoch_loss = epoch_loss / num_batches
    avg_epoch_accuracy = epoch_correct / len(data_loader.dataset)

    logger.info(f"Epoch {epoch + 1} finished - Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_epoch_accuracy:.4f}")

    return avg_epoch_loss, avg_epoch_accuracy


def train_model_sync(load_train, sample_rate, num_samples, epochs, batch_size, logger):
    """Синхронна функція тренування"""

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Створюємо директорії
        target_dir = Path(variables.file_dir) / load_train.name
        if target_dir.is_dir():
            logger.info(f"Cleaning target directory {target_dir}, preserving logs...")
            for item in target_dir.iterdir():
                if item.name == "logs":
                    continue  # Пропускаємо папку з логами, яку ми щойно створили в async функції

                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete item {item}: {e}")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)

        # Mel spectrogram transform
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )

        logger.info("Creating dataset CSV...")
        # Створюємо CSV (використовуємо SessionLocal для нового з'єднання)
        db = SessionLocal()
        try:
            create_csv(db, load_train, False)
        finally:
            db.close()

        csv_filename = f"dataset_{datetime.datetime.utcnow().strftime('%Y_%m_%d')}.csv"
        csv_filepath = os.path.join(variables.file_dir, load_train.name, csv_filename)

        audio_dir = os.path.join(variables.file_dir, load_train.name, "audio")
        Path(audio_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Dataset CSV filename is located in {csv_filepath}")

        # ⭐ КРИТИЧНО: Завантажуємо CSV для створення class_mapping
        df = pd.read_csv(csv_filepath)

        # ⭐ Створюємо правильний class_mapping на основі target + category
        # Групуємо по target та category, щоб отримати унікальні комбінації
        unique_classes = df[['target', 'category']].drop_duplicates().sort_values('target')

        num_classes = len(unique_classes)
        logger.info(f"Detected {num_classes} classes for training.")

        # Створюємо маппінг: індекс (target) -> назва класу (category)
        class_mapping = {
            int(row['target']): row['category']
            for _, row in unique_classes.iterrows()
        }

        logger.info(f"Created class mapping: {class_mapping}")

        # ⭐ ЗБЕРІГАЄМО class_mapping для майбутнього використання
        mapping_file = target_dir / "class_mapping.json"
        save_class_mapping(class_mapping, str(mapping_file))
        logger.info(f"Saved class mapping to {mapping_file}")

        logger.info("Loading dataset...")

        # Завантажуємо dataset
        usd = VoipTimeDataset(
            csv_filepath,
            audio_dir,
            mel_spectrogram,
            sample_rate,
            num_samples,
            device
        )

        logger.info(f"Dataset loaded with {len(usd)} samples")

        # Отримуємо мітки
        labels_target = usd.annotations["target"]
        labels_unique, counts = np.unique(labels_target, return_counts=True)

        logger.info(f"Unique target labels: {labels_unique}")
        logger.info(f"Label counts: {dict(zip(labels_unique, counts))}")

        # Розраховуємо ваги для класів
        total_samples = len(labels_target)
        class_weights_values = [total_samples / count if count > 0 else 0 for count in counts]
        class_weights_dict = {label: weight for label, weight in zip(labels_unique, class_weights_values)}

        logger.info(f"Class weights: {class_weights_dict}")

        # Призначаємо ваги для кожного прикладу
        example_weights = [class_weights_dict[label] for label in labels_target]
        logger.info(f"Example weights sample: {example_weights[:5]}")

        # Створюємо sampler і dataloader
        sampler = WeightedRandomSampler(example_weights, len(labels_target))
        train_dataloader = DataLoader(usd, batch_size=batch_size, sampler=sampler)

        logger.info(f"DataLoader created with batch_size={batch_size}, {len(train_dataloader)} batches")

        # Створюємо модель
        logger.info("Initializing model...")
        cnn = CNNNetwork(num_classes=num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(cnn.parameters(), lr=0.0001)

        # TensorBoard
        tensor_board_log = os.path.join(variables.file_dir, load_train.name, "tensorboard")
        Path(tensor_board_log).mkdir(parents=True, exist_ok=True)
        board_writer = SummaryWriter(log_dir=tensor_board_log)

        logger.info(f"Starting training for {epochs} epochs...")

        torchaudio.set_audio_backend("ffmpeg")

        # Тренування
        for epoch in range(epochs):
            logger.info(f"Starting Epoch {epoch + 1}/{epochs}")
            avg_loss, avg_acc = train_single_epoch(
                cnn, train_dataloader, loss_fn, optimiser, device, epoch, logger
            )

            # TensorBoard logging
            board_writer.add_scalar("epoch training loss", avg_loss, epoch)
            board_writer.add_scalar("epoch accuracy", avg_acc, epoch)

            logger.info("---------------------------")

        logger.info("Finished training successfully.")

        # Зберігаємо модель
        logger.info("Saving trained model...")
        torch_filepath = os.path.join(variables.file_dir, load_train.name, f"{load_train.name}.pth")
        torch.save(cnn.state_dict(), torch_filepath)
        logger.info(f"Model saved to {torch_filepath}")

        board_writer.close()
        logger.info("Training task completed successfully.")

        return True

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise


async def run_training_task(ctx, train_id: int, sample_rate: int, num_samples: int, epochs: int, batch_size: int):
    """Основна функція тренування з логуванням в реальному часі"""

    print(f"\n{'=' * 80}")
    print(f"[TASK START] Starting training task for train_id={train_id}")
    print(
        f"[TASK START] Parameters: sample_rate={sample_rate}, num_samples={num_samples}, epochs={epochs}, batch_size={batch_size}")
    print(f"{'=' * 80}\n")

    db = SessionLocal()
    redis_channel = f"train_logs:{train_id}"

    log_queue = Queue()
    stop_event = asyncio.Event()

    # Налаштування логера
    logger = logging.getLogger(f"train_{train_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    # ⭐ Зберігаємо посилання на handlers окремо
    redis_handler = None
    file_handler = None
    log_file_path = None  # ⭐ Додаємо для доступу в finally

    try:
        print(f"[TASK] Loading train from database...")
        load_train = load_train_by_id(db, train_id)

        if not load_train:
            print(f"[TASK ERROR] Train ID {train_id} not found in database!")
            return

        print(f"[TASK] Train loaded: name='{load_train.name}', status={load_train.status}")
        load_train = load_train_by_id(db, train_id)
        if not load_train:
            print(f"[TASK] Train ID {train_id} not found")
            return

        # ⭐ ДІАГНОСТИКА: Перевіряємо базову директорію
        base_dir = Path(variables.file_dir)
        print(f"[TASK] Base directory: {base_dir}, exists: {base_dir.exists()}")

        # Створюємо директорію для логів
        log_dir = Path(variables.file_dir) / load_train.name / "logs"
        print(f"[TASK] Creating log directory: {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TASK] Log directory created successfully, exists: {log_dir.exists()}")

        # Повний шлях до файлу логів
        log_file_path = log_dir / f"{load_train.name}.log"
        print(f"[TASK] Log file path: {log_file_path}")

        # Додаємо Redis handler
        print(f"[TASK] Creating Redis handler...")
        redis_handler = RedisLogHandler(log_queue)
        redis_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(redis_handler)
        print(f"[TASK] ✅ Redis handler added")

        # Додаємо файловий handler
        print(f"[TASK] Creating file handler for: {log_file_path}")
        try:
            file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(file_handler)
            print(f"[TASK] ✅ File handler created and added to logger")
        except Exception as fh_err:
            print(f"[TASK] ❌ FAILED to create file handler: {fh_err}")
            raise

        # Запускаємо publisher
        print(f"[TASK] Starting publisher for channel {redis_channel}")
        publisher_task = asyncio.create_task(
            redis_log_publisher(redis_channel, log_queue, stop_event)
        )

        await asyncio.sleep(0.5)

        # ⭐ ТЕСТУЄМО логування одразу після створення handler
        logger.info("=" * 60)
        logger.info(f"Starting training task for Train ID: {train_id}, Name: {load_train.name}")
        logger.info(
            f"Parameters: sample_rate={sample_rate}, num_samples={num_samples}, epochs={epochs}, batch_size={batch_size}")
        logger.info("=" * 60)

        # ⭐ ФОРСУЄМО запис у файл одразу
        if file_handler:
            file_handler.flush()
            print(f"[TASK] Initial log flushed to file")

        # ⭐ ПЕРЕВІРЯЄМО, чи з'явився файл
        if log_file_path.exists():
            file_size = log_file_path.stat().st_size
            print(f"[TASK] Log file exists! Size: {file_size} bytes")
        else:
            print(f"[TASK] WARNING: Log file does NOT exist after initial write!")

        # Оновлюємо статус
        load_train.status = TrainStatus.RUNNING
        update_train(db, load_train)
        logger.info("Updated train status to RUNNING")

        # Викликаємо синхронне тренування в executor
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            train_model_sync,
            load_train,
            sample_rate,
            num_samples,
            epochs,
            batch_size,
            logger
        )

        if success:
            load_train.status = TrainStatus.COMPLETED
            logger.info("Training completed successfully. Status updated to COMPLETED.")
        else:
            load_train.status = TrainStatus.FAILED
            logger.error("Training failed. Status updated to FAILED.")

        update_train(db, load_train)

    except Exception as e:
        print(f"[TASK] Training failed with exception: {e}")
        logger.error(f"Training failed: {e}", exc_info=True)

        try:
            load_train = load_train_by_id(db, train_id)
            if load_train:
                load_train.status = TrainStatus.FAILED
                update_train(db, load_train)
        except Exception as update_err:
            print(f"[TASK] Failed to update status: {update_err}")

    finally:
        # ⭐ КРИТИЧНО: Спочатку flush файловий handler
        print(f"[TASK] Flushing file logs...")
        if file_handler:
            try:
                file_handler.flush()
                print(f"[TASK] ✅ File handler flushed successfully")

                # ⭐ Перевіряємо фінальний стан файлу
                if log_file_path and log_file_path.exists():
                    file_size = log_file_path.stat().st_size
                    print(f"[TASK] ✅ Log file exists! Size: {file_size} bytes, Path: {log_file_path}")
                else:
                    print(f"[TASK] ⚠️ WARNING: Log file does NOT exist: {log_file_path}")
            except Exception as e:
                print(f"[TASK] ❌ Error flushing file handler: {e}")

        # Логуємо завершення
        logger.info("Training task finished. Closing log handlers...")

        # ⭐ Ще один flush після останнього повідомлення
        if file_handler:
            file_handler.flush()

        # Сигналізуємо publisher про завершення
        print(f"[TASK] Signaling log publisher to finish...")
        stop_event.set()

        # Чекаємо, поки publisher опублікує всі логи
        print(f"[TASK] Waiting for log queue to empty...")
        await asyncio.sleep(1)

        # Для синхронної Queue використовуємо join() в executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, log_queue.join)

        print(f"[TASK] Waiting for publisher task to complete...")
        await publisher_task

        # ⭐ ВАЖЛИВО: Закриваємо handlers правильно
        print(f"[TASK] Closing handlers...")

        # Видаляємо handlers з logger
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Закриваємо тільки файловий handler (у нього є файловий дескриптор)
        if file_handler:
            try:
                file_handler.flush()  # Останній flush
                file_handler.close()
                print(f"[TASK] ✅ File handler closed successfully")
            except Exception as e:
                print(f"[TASK] ❌ Error closing file handler: {e}")

        # ⭐ ФІНАЛЬНА ПЕРЕВІРКА: Чи існує файл після закриття?
        if log_file_path:
            if log_file_path.exists():
                final_size = log_file_path.stat().st_size
                print(f"[TASK] ✅✅✅ SUCCESS! Log file saved: {log_file_path}")
                print(f"[TASK] Final file size: {final_size} bytes")
            else:
                print(f"[TASK] ❌❌❌ CRITICAL: Log file DISAPPEARED after closing: {log_file_path}")

        # Redis handler не потребує close (він не має файлового дескриптора)
        if redis_handler:
            try:
                # Просто видаляємо посилання
                del redis_handler
            except Exception as e:
                print(f"[TASK] Error removing redis handler: {e}")

        db.close()
        print(f"[TASK] Task cleanup completed for train_id={train_id}")
        print(f"{'=' * 80}\n")