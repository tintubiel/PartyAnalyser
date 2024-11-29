import requests
from ultralytics import YOLO
from omegaconf import OmegaConf
from PIL import Image
from src.services.constants import CONFIG_PATH
from io import BytesIO
import numpy as np

class PizzaPepperoniDetector:
    def __init__(self, config: dict = None):
        """Инициализация детектора.

        :param config: Конфигурация загрузки модели.
        """

        if config is None: # Загрузка конфигурации из файла
            cfg = OmegaConf.load(CONFIG_PATH)
            config = cfg['services']['pizza_detection']
        else:
            config = config

        # Загрузка модели
        self.model = YOLO(config['model_path'])
        self.pizza_class_id = config['pizza_class_id']
        self.pepperoni_class_id = config['pepperoni_class_id']

    def detect_from_image(self, image: np.ndarray) -> (bool, bool):
        """Детекция пиццы и пеперони на изображении из пути к файлу.

        :param image: Изображение в виде массива;
        :return: Кортеж (pizza_found, pepperoni_found).
        """
        results = self.model.predict(image, verbose=False)

        return self._process_detection_results(results)

    def detect_from_image_path(self, image_path: str) -> (bool, bool):
        """Детекция пиццы и пеперони на изображении из пути к файлу.

        :param image_path: Путь к изображению;
        :return: Кортеж (pizza_found, pepperoni_found).
        """
        results = self.model.predict(image_path, verbose=False)

        return self._process_detection_results(results)

    def detect_from_url(self, url: str) -> (bool, bool):
        """Детекция пиццы и пеперони на изображении из URL.

        :param url: URL изображения;
        :return: Кортеж (pizza_found, pepperoni_found).
        """
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        results = self.model.predict(image, verbose=False)  # Предполагается, что этот метод также принимает PIL Image.

        return self._process_detection_results(results)

    def detect_image_proba(self, image_path: str) -> (float, float):
        """Максимальная вероятность нахождения пиццы и пепперони на изображении.

        :param image: Изображение в виде массива;
        :return: Кортеж (pizza_probability, pepperoni_probability).
        """
        results = self.model.predict(image_path)

        for result in results:

            # Инициализируем вероятности
            pizza_probability = float('-inf')
            pepperoni_probability = float('-inf')
            # Итерируемся по всем обнаруженным объектам
            for box in result.boxes:
                # Получаем класс объекта и его коэффициент уверенности
                class_id = int(box.cls)  # Индекс класса
                confidence = box.conf.item()*100

                # Проверяем, является ли это пиццей или пепперони
                if class_id == self.pizza_class_id:
                    pizza_probability = max(pizza_probability, confidence)
                elif class_id == self.pepperoni_class_id:
                    pepperoni_probability = max(pepperoni_probability, confidence)

        return max(pizza_probability, 0.0), max(pepperoni_probability, 0.0)

    def _process_detection_results(self, results) -> (bool, bool):
        """Обработка результатов детекции.

        :param results: Результаты предсказания модели.
        :return: Кортеж (pizza_found, pepperoni_found).
        """
        pizza_found = False
        pepperoni_found = False

        # Обработка результатов
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id == self.pizza_class_id:
                    pizza_found = True
                elif class_id == self.pepperoni_class_id:
                    pepperoni_found = True

        return pizza_found, pepperoni_found


if __name__ == "__main__":
    pass
