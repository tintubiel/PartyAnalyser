from fastapi import FastAPI
from omegaconf import OmegaConf
from src.containers.conainers import AppContainer
from src.routes.routers import router as app_router
import uvicorn
from src.routes import analyse as party_routes


def create_app() -> FastAPI:
    container = AppContainer()
    cfg = OmegaConf.load('configs/config.yaml')
    container.config.from_dict(cfg)
    container.wire([party_routes])

    app = FastAPI()
    app.include_router(app_router)
    return app


if __name__ == '__main__':
    app = create_app()
    uvicorn.run(app, port=2444, host='0.0.0.0')
