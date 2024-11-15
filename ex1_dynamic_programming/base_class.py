from abc import ABC, abstractmethod


class Env(ABC):
    def __init__(self):
        self._case = None
        self._h = None # to be specified in son class
        self._theta = None # to be specified in son class
        self._target = None
        self._constraint = None

    def set_case(self, case):
        self._case = case

    def get_case(self):
        return self._case

    def set_target(self, target):
        self._target = target

    def get_target(self):
        return self._target
    
class Controller(ABC):

    def __init__(self):
        self.input = None
        self._target = None

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = value
    
    @abstractmethod
    def test_dynamics():
        pass