"""
Утиліти для роботи з маппінгом класів у моделях класифікації.
"""
import json
from pathlib import Path
from typing import Dict, List
import os


def create_class_mapping_from_directory(audio_dir: str) -> Dict[int, str]:
    """
    Створює маппінг класів на основі структури директорій.
    Припускаємо, що аудіофайли розташовані в піддиректоріях,
    де назва піддиректорії = назва класу.

    Args:
        audio_dir: Шлях до директорії з аудіофайлами

    Returns:
        Словник з маппінгом індексів до назв класів
    """
    audio_path = Path(audio_dir)

    if not audio_path.exists():
        raise FileNotFoundError(f"Directory not found: {audio_dir}")

    # Отримуємо всі піддиректорії (класи)
    class_names = sorted([
        d.name for d in audio_path.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    # Створюємо маппінг (індекси відповідають алфавітному порядку)
    class_mapping = {idx: name for idx, name in enumerate(class_names)}

    return class_mapping


def save_class_mapping(class_mapping: Dict[int, str], save_path: str) -> None:
    """
    Зберігає маппінг класів у JSON файл.

    Args:
        class_mapping: Словник з маппінгом класів
        save_path: Шлях для збереження файлу
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=2)


def load_class_mapping(mapping_path: str) -> Dict[int, str]:
    """
    Завантажує маппінг класів з JSON файлу.

    Args:
        mapping_path: Шлях до файлу з маппінгом

    Returns:
        Словник з маппінгом класів
    """
    mapping_file = Path(mapping_path)

    if not mapping_file.exists():
        raise FileNotFoundError(f"Class mapping file not found: {mapping_path}")

    with open(mapping_file, 'r', encoding='utf-8') as f:
        # JSON зберігає ключі як strings, конвертуємо в int
        mapping_data = json.load(f)
        class_mapping = {int(k): v for k, v in mapping_data.items()}

    return class_mapping


def get_class_mapping_for_train(train_name: str, base_dir: str) -> Dict[int, str]:
    """
    Отримує маппінг класів для конкретного тренування.
    Спочатку намагається завантажити з файлу, якщо файл не існує -
    створює на основі структури директорій.

    Args:
        train_name: Назва тренування
        base_dir: Базова директорія з файлами

    Returns:
        Словник з маппінгом класів
    """
    train_dir = Path(base_dir) / train_name
    mapping_file = train_dir / "class_mapping.json"
    audio_dir = train_dir / "audio"

    # Спочатку намагаємося завантажити існуючий маппінг
    if mapping_file.exists():
        try:
            return load_class_mapping(str(mapping_file))
        except Exception as e:
            print(f"Warning: Could not load existing mapping: {e}")

    # Якщо файлу немає, створюємо новий маппінг
    if audio_dir.exists():
        class_mapping = create_class_mapping_from_directory(str(audio_dir))
        # Зберігаємо для майбутнього використання
        save_class_mapping(class_mapping, str(mapping_file))
        return class_mapping

    raise FileNotFoundError(f"Neither mapping file nor audio directory found for train: {train_name}")


def get_class_names(class_mapping: Dict[int, str]) -> List[str]:
    """
    Повертає список назв класів у правильному порядку.

    Args:
        class_mapping: Словник з маппінгом класів

    Returns:
        Список назв класів
    """
    max_idx = max(class_mapping.keys())
    return [class_mapping[i] for i in range(max_idx + 1)]


def get_num_classes(class_mapping: Dict[int, str]) -> int:
    """
    Повертає кількість класів.

    Args:
        class_mapping: Словник з маппінгом класів

    Returns:
        Кількість класів
    """
    return len(class_mapping)


# Приклад використання:
if __name__ == "__main__":
    # Створення маппінгу з директорії
    audio_dir = "/path/to/audio/files"
    mapping = create_class_mapping_from_directory(audio_dir)
    print("Class mapping:", mapping)

    # Збереження маппінгу
    save_class_mapping(mapping, "/path/to/save/class_mapping.json")

    # Завантаження маппінгу
    loaded_mapping = load_class_mapping("/path/to/save/class_mapping.json")
    print("Loaded mapping:", loaded_mapping)

    # Отримання інформації про класи
    print("Class names:", get_class_names(mapping))
    print("Number of classes:", get_num_classes(mapping))