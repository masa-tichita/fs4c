from pydantic import BaseModel

from utils.logging import log_fn, setup_logger
from utils.system import project_root

setup_logger()


# TODO: 挙動の確認のためのスクリプトあとで消す
@log_fn
class User(BaseModel):
    id: int
    name: str
    is_active: bool


user = User(id=1, name="John", is_active=True)
print(user.model_dump())
print(project_root())
