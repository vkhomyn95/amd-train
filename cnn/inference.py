import torch
import torchaudio
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np

from cnn.cnn import CNNNetwork


class ModelInference:
    """
    Клас для виконання інференсу (передбачення) на навчених моделях аудіокласифікації.
    """

    def __init__(
            self,
            model_path: str,
            class_mapping: Dict[int, str],
            sample_rate: int = 16000,
            num_samples: int = 22050,
            device: str = None
    ):
        """
        Ініціалізація інференс-моделі.

        Args:
            model_path: Шлях до збереженої моделі (.pth файл)
            class_mapping: Словник з маппінгом індексів класів до їх назв {0: "human", 1: "voicemail"}
            sample_rate: Частота дискретизації (має співпадати з тренувальною)
            num_samples: Кількість семплів для обробки
            device: Пристрій для обчислень ("cuda" або "cpu")
        """
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.class_mapping = class_mapping

        # Визначаємо пристрій
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("Testing device is:", self.device)

        # Ініціалізуємо Mel Spectrogram transform (такий самий як при тренуванні)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        ).to(self.device)

        # Завантажуємо модель
        self.model = self._load_model(model_path)
        self.model.eval()  # Переводимо в режим оцінки

    def _load_model(self, model_path: str) -> CNNNetwork:
        """
        Завантажує навчену модель з файлу.

        Args:
            model_path: Шлях до файлу моделі

        Returns:
            Завантажена модель
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = CNNNetwork().to(self.device)

        # Завантажуємо ваги моделі
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)

        return model

    def _preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Передобробка аудіофайлу для інференсу.

        Args:
            audio_path: Шлях до аудіофайлу

        Returns:
            Тензор з preprocessed аудіо
        """
        # Завантажуємо аудіо
        signal, sr = torchaudio.load(audio_path)

        # Конвертуємо в моно, якщо стерео
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        # Resampling, якщо потрібно
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal = resampler(signal)

        # Обрізаємо або доповнюємо до потрібної довжини
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        elif signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)

        # Переносимо на пристрій
        signal = signal.to(self.device)

        # Застосовуємо Mel Spectrogram
        signal = self.mel_spectrogram(signal)

        return signal

    def predict(self, audio_path: str) -> Tuple[int, str, float]:
        """
        Виконує передбачення для одного аудіофайлу.

        Args:
            audio_path: Шлях до аудіофайлу

        Returns:
            Кортеж: (індекс класу, назва класу, впевненість)
        """
        # Передобробка
        signal = self._preprocess_audio(audio_path)

        # Додаємо batch dimension
        signal = signal.unsqueeze(0)

        # Виконуємо інференс
        with torch.no_grad():
            predictions = self.model(signal)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)
            predicted_index = predictions.argmax(1).item()
            confidence = probabilities[0][predicted_index].item()

        # Отримуємо назву класу
        predicted_label = self.class_mapping.get(predicted_index, f"Unknown_{predicted_index}")

        return predicted_index, predicted_label, confidence

    def predict_batch(self, audio_paths: List[str]) -> List[Dict]:
        """
        Виконує передбачення для багатьох файлів.

        Args:
            audio_paths: Список шляхів до аудіофайлів

        Returns:
            Список словників з результатами
        """
        results = []

        for audio_path in audio_paths:
            try:
                class_idx, label, confidence = self.predict(audio_path)
                results.append({
                    "filename": Path(audio_path).name,
                    "class_index": class_idx,
                    "label": label,
                    "confidence": confidence,
                    "success": True,
                    "error": None
                })
            except Exception as e:
                results.append({
                    "filename": Path(audio_path).name,
                    "class_index": None,
                    "label": None,
                    "confidence": None,
                    "success": False,
                    "error": str(e)
                })

        return results

    def get_class_probabilities(self, audio_path: str) -> Dict[str, float]:
        """
        Повертає ймовірності для всіх класів.

        Args:
            audio_path: Шлях до аудіофайлу

        Returns:
            Словник з ймовірностями для кожного класу
        """
        signal = self._preprocess_audio(audio_path)
        signal = signal.unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(signal)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)

        result = {}
        for idx, prob in enumerate(probabilities[0]):
            class_name = self.class_mapping.get(idx, f"Class_{idx}")
            result[class_name] = prob.item()

        return result