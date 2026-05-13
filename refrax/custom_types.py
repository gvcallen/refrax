from typing import TypeVar, Literal, Any

TRoot = TypeVar("TRoot")
PathOp = Literal["attr", "item"]
PathStep = tuple[PathOp, Any]