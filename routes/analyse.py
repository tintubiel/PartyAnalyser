import os

import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File

from src.containers.conainers import AppContainer
from src.routes.routers import router
from src.services.party_analyser import PizzaWineAnalytics


@router.get("/")
def read_root():
    return {"message": "Добро пожаловать в приложение FastAPI!"}


@router.get('/answers')
@inject
def answers_list(service: PizzaWineAnalytics = Depends(Provide[AppContainer.party_analyser])):
    return {
        'answers': service.answers
    }


@router.post('/predict')
@inject
def predict(
        image: bytes = File(),
        service: PizzaWineAnalytics = Depends(Provide[AppContainer.party_analyser]),
):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    result = service.predict(img)

    return {'answer': result}


