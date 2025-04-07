from fastapi import HTTPException

from .auth import router
from .train import router
from .dataset import router
from .user import router
from .label import router


class CustomHTTPException(HTTPException):
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)
        self.success = False
        self.message = detail
