from abc import ABC, abstractmethod
from copy import deepcopy

class IRepository[T](ABC):
    @abstractmethod
    def get(id: str) -> T:
        pass

    @abstractmethod
    def get_all() -> list[T]:
        pass

    @abstractmethod
    def create(id: str, entity: T):
        pass

    @abstractmethod
    def update(id: str, entity: T):
        pass

    @abstractmethod
    def delete(id: str):
        pass


class InMemoryRepository[T](IRepository):
    def __init__(self):
        super().__init__()
        self.data = {}
    
    def create(self, id: str, entity: T):
        self.data[id] = deepcopy(entity)

    def get(self, id):
        return deepcopy(self.data.get(id, None))

    def get_all(self) -> list[T]:
        return deepcopy(list(self.data.values()))

    def update(self, id: str, entity: T):
        self.data[id] = deepcopy(entity)

    def delete(self, id: str):
        self.data.pop(id, "")