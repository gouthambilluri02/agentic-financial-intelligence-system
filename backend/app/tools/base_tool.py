from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """
    Common interface implemented by every agent tool.

    Each tool must provide:
    - a unique name
    - a description
    - a run() method
    """

    name: str
    description: str

    @abstractmethod
    def run(
        self,
        **kwargs: Any,
    ) -> dict:
        """
        Execute the tool and return a structured result.
        """

        raise NotImplementedError

    def get_metadata(self) -> dict[str, str]:
        """
        Return information that the agent can use when selecting tools.
        """

        return {
            "name": self.name,
            "description": self.description,
        }