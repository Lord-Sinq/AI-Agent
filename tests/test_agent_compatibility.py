import unittest

from agent import Agent as AgentFromAgent
from base_agent import Agent as AgentFromBase


class AgentCompatibilityTest(unittest.TestCase):
    def test_agent_module_reuses_shared_base_class(self):
        self.assertIs(AgentFromAgent, AgentFromBase)


if __name__ == "__main__":
    unittest.main()
