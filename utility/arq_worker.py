from arq.connections import RedisSettings
from utility.tasks import run_training_task

REDIS_HOST = 'localhost'
REDIS_PORT = 6379


async def startup(ctx):
    print("ARQ Worker starting up...")


async def shutdown(ctx):
    print("ARQ Worker shutting down...")


class WorkerSettings:
    functions = [run_training_task]
    redis_settings = RedisSettings(host=REDIS_HOST, port=REDIS_PORT)
    on_startup = startup
    on_shutdown = shutdown
    keep_result = -1
