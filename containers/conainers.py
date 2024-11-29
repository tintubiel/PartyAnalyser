from dependency_injector import containers, providers

from src.services.party_analyser import PizzaWineAnalytics
from src.services.wine_classification import WineClassificator
from src.services.pizza_detection import PizzaPepperoniDetector


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    wine_classificator = providers.Singleton(
        WineClassificator,
        config=config.services.wine_classification,
    )

    pizza_detector = providers.Singleton(
        PizzaPepperoniDetector,
        config=config.services.pizza_detection,
    )

    party_analyser = providers.Singleton(
        PizzaWineAnalytics,
        classification_model=wine_classificator,
        detection_model=pizza_detector,
    )
