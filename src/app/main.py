import logging
import os

from api import router
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.datastructures import CommaSeparatedStrings
from starlette.middleware.cors import CORSMiddleware
from starlette_context.middleware import RawContextMiddleware

API_PORT = 8081
ROOT_PATH = os.getenv("ROOT_PATH", "")
API_DEBUG = True
ALLOWED_HOSTS = CommaSeparatedStrings(os.getenv("ALLOWED_HOSTS", "*"))
LOG_LEVEL = os.getenv("LOG_LEVEL")
OPEN_API_VERSION = "2.1.0"
logger = logging.getLogger(f"{os.getenv('LOGGER', 'gunicorn')}.error")


def api() -> FastAPI:
    _api = FastAPI(
        root_path=ROOT_PATH,
        title="Meta Yovisto API",
        version=OPEN_API_VERSION,
        debug=API_DEBUG,
    )
    logger.debug(f"Launching FastAPI on root path {ROOT_PATH}")

    _api.include_router(router)

    for route in _api.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name

    _api.add_middleware(RawContextMiddleware)

    return _api


load_dotenv()

app = CORSMiddleware(
    app=api(),
    allow_origins=ALLOWED_HOSTS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count"],
)

if __name__ == "__main__":
    import os

    import uvicorn

    conf = {
        "host": "0.0.0.0",
        "port": API_PORT,
        "reload": True,
        "reload_dirs": [f"{os.getcwd()}"],
        "log_level": LOG_LEVEL,
    }

    uvicorn.run("app.main:app", **conf)
