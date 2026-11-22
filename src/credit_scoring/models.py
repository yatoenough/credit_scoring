from pydantic import BaseModel, Field


class ClientData(BaseModel):
    age: int = Field(ge=21, le=100)
    income: float = Field(gt=0)
    education: bool
    work: bool
    car: bool
