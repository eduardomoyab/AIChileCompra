from pydantic import BaseModel

# Estado en memoria
_token_store: dict = {}

class TokenPayload(BaseModel):
    access_token: str
    obtained_at: str