# ------------------------------------------------------------------------------
# An agent that tracks accuracy.
# ------------------------------------------------------------------------------

from ptt.agents.agent import Agent
from ptt.eval.metrics import accuracy

class ClassificationAgent(Agent):
    def __init__(self, config, base_criterion, verbose):
        super().__init__(config, base_criterion, verbose)
        self.metrics['accuracy'] = accuracy