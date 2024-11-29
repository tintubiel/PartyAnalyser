from typing import Dict, Tuple
import cv2
import numpy as np
from PIL import Image
from src.services.constants import CONFIG_PATH
from src.services.pizza_detection import PizzaPepperoniDetector
from src.services.wine_classification import WineClassificator
from omegaconf import OmegaConf


class PizzaWineAnalytics:
    def __init__(self, detection_model: PizzaPepperoniDetector, classification_model: WineClassificator):

        self.detection_model = detection_model
        self.classification_model = classification_model

    @property
    def answers(self):
        return {
            "Пиццы нет, идти на вечеринку не стоит.",
            "На столе есть пицца и вино! Иди на вечеринку!",
            "Пицца есть, но колбасы нет, значит, на вечеринку не стоит идти!",
            "Пицца с колбасой! Иди на вечеринку!"}

    def predict(self, image: np.ndarray) -> Dict[str, bool]:
        """Предсказание наличия пиццы, пеперони и вина по изображению.

        :param image: входное RGB изображение;
        :return: словарь вида {'pizza': True/False, 'pepperoni': True/False, 'wine': True/False/None}.
        """

        # Проверка наличия пиццы и пеперони
        pizza_found, pepperoni_found = self.detection_model.detect_from_image(image)

        # Если пеперони не обнаружена, классифицируем наличие вина
        wine_found = None
        if not pepperoni_found:
            wine_found = self.classification_model.classificate_from_image(image)

        results = {
            'pizza': pizza_found,
            'pepperoni': pepperoni_found,
            'wine': wine_found
        }

        return self._process_results(results)


    def _process_results(self, results) -> str:
        """Обработка результатов.

        :param results: Результаты предсказания модели.
        :return: Строка с описанием решения о вечеринке.
        """

        pizza_found = results.get('pizza', False)
        pepperoni_found = results.get('pepperoni', False)
        wine_found = results.get('wine', False)

        # Проверяем наличие пиццы
        if not pizza_found:
            return "Пиццы нет, идти на вечеринку не стоит."
        if wine_found:
            return "На столе есть пицца и вино! Иди на вечеринку!"
        if not pepperoni_found:
            return "Пицца есть, но колбасы нет, значит, на вечеринку не стоит идти!"
        return "Пицца с колбасой! Иди на вечеринку!"


if __name__ == "__main__":
    classificator = WineClassificator()
    detector = PizzaPepperoniDetector()

    analyser = PizzaWineAnalytics(detector, classificator)
    image_path = '/Users/tintubiel/Desktop/img-20181208-205558-largejpg.jpg'
    image = Image.open(image_path)
    image = np.array(image)
    print(analyser.predict(image))
