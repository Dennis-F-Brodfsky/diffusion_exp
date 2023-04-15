import abc


class BaseCritic(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def qa_values(self, obs, **kwargs):
        raise NotImplementedError


class BasePolicy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action(self, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, obs, action, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, filepath: str):
        raise NotImplementedError

    def set_critic(self, critic):
        self.__setattr__('critic', critic)


class BaseAgent(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self, **kwargs) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def add_to_replay_buffer(self, paths, add_noised=False):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path):
        raise NotImplementedError


class BaseModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        raise NotImplementedError

    @abc.abstractmethod
    def get_prediction(self, ob_no, ac_na, data_statistics):
        raise NotImplementedError


class Schedule(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def value(self, t):
        raise NotImplementedError
