import cv2
import requests
import torch
from io import BytesIO
from PIL import Image
from omegaconf import OmegaConf
import torchvision.models as models
from src.services.models.lightning_module import LModel
from src.services.preprocess_utils import preprocess_image
from src.services.constants import CONFIG_PATH
import torch.nn.functional as F
import numpy as np

class WineClassificator:
    def __init__(self, config: dict = None):
        """Инициализация классификатора.

        :param config: Конфигурация загрузки модели.
        """

        if config is None: # Загрузка конфигурации из файла
            cfg = OmegaConf.load(CONFIG_PATH)
            config = cfg['services']['wine_classification']
        else:
            config = config

        # Загрузка модели
        self.model = LModel.load_from_checkpoint(config['model_path'])
        self.not_wine_class_id = config['not_wine_class_id']
        self.wine_bottles_class_id = config['wine_bottles_class_id']
        self.wine_glasses_class_id = config['wine_glasses_class_id']

    def classificate_from_image(self, image: np.ndarray) -> (bool, bool):
        """Классификация наличия вина на изображении из пути к файлу.

        :param image_path: Путь к изображению;
        :return: Кортеж (pizza_found, pepperoni_found).
        """
        preprocessed_image = preprocess_image(image, target_image_size=(224, 224))
        # results = self.model.predict(preprocessed_image)
        with torch.no_grad():  # Отключаем градиенты для оптимизации
            results = self.model(preprocessed_image)  # Получаем предсказания
            # Применение функции softmax для получения вероятностей
            probabilities = F.softmax(results, dim=1)
            # Получение класса с максимальной вероятностью
            predicted_class = torch.argmax(probabilities, dim=1).item()

            # print("Вероятности классов:", probabilities)
            # print("Предсказанный класс:", predicted_class.item())
            # print({'not_wine': 0, 'wine_bottles': 1, 'wine_glasses': 2})
        return self._process_classification_results(predicted_class)


    def classificate_from_image_path(self, image_path: str) -> (bool, bool):
        """Классификация наличия вина на изображении из пути к файлу.

        :param image_path: Путь к изображению;
        :return: Кортеж (pizza_found, pepperoni_found).
        """
        image = cv2.imread(image_path)
        preprocessed_image = preprocess_image(image, target_image_size=(224, 224))
        # results = self.model.predict(preprocessed_image)
        with torch.no_grad():  # Отключаем градиенты для оптимизации
            results = self.model(preprocessed_image)  # Получаем предсказания
            probabilities = F.softmax(results, dim=1) # Применение функции softmax для получения вероятностей
            predicted_class = torch.argmax(probabilities, dim=1).item()# Получение класса с максимальной вероятностью

            # print("Вероятности классов:", probabilities)
            # print("Предсказанный класс:", predicted_class.item())
            # print({'not_wine': 0, 'wine_bottles': 1, 'wine_glasses': 2})
        return self._process_classification_results(predicted_class)

    def classificate_from_url(self, url: str) -> (bool, bool):
        """Классификация наличия вина на изображении из URL.

        :param url: URL изображения;
        :return: Кортеж (pizza_found, pepperoni_found).
        """
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        results = self.model.predict(image)

        return self._process_classification_results(results)

    def _process_classification_results(self, results) -> bool:
        """Обработка результатов классификации.

        :param results: Результаты предсказания модели.
        :return: Кортеж (bottle_found, glass_found).
        """
        wine_found = False
        if results > 0:
            wine_found = True
        return wine_found

if __name__ == "__main__":
    pass
