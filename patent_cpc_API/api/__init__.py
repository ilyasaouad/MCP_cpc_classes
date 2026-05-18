"""api package — holds engine and KG references set at startup."""

_engine = None
_kg = None


def set_engine(engine) -> None:
    global _engine
    _engine = engine


def set_kg(kg) -> None:
    global _kg
    _kg = kg
