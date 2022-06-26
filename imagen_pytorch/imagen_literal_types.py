from typing import Literal
_LossType = Literal['l1', 'l2', 'huber']
_NoiseSchedule = Literal['cosine', 'linear']
_PredObjective = Literal['noise']