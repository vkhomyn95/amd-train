import asyncio
from cProfile import label

import uvicorn
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from database import get_db, insert_default_user
from database.crud import insert_default_roles, insert_default_notation
from logger.logger import Logger
from routes import CustomHTTPException, auth, train, user, dataset, label
from variables import variables

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

app.mount("/static", StaticFiles(directory=variables.base_dir + "/static"), name="static")

templates = Jinja2Templates(directory=variables.base_dir + "/templates")

logger = Logger(variables.logger_dir)


@app.exception_handler(CustomHTTPException)
async def custom_http_exception_handler(request: Request, exc: CustomHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": exc.success, "message": exc.message}
    )


app.include_router(auth.router)

app.include_router(train.router, prefix="/trains")

app.include_router(dataset.router, prefix="/datasets")

app.include_router(user.router, prefix="/users")

app.include_router(label.router, prefix="/labels")

db = next(get_db())

insert_default_roles(db)

insert_default_user(db)

insert_default_notation(db)


async def start_fastapi():
    config = uvicorn.Config(app, host=variables.app_host, port=variables.app_port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def run_servers():
    fastapi_task = asyncio.create_task(start_fastapi())
    await asyncio.gather(fastapi_task)


if __name__ == "__main__":
    try:
        asyncio.run(run_servers())
    except (KeyboardInterrupt, SystemExit):
        pass
