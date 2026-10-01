from abc import ABC, abstractmethod


class BaseVLMEvaluator(ABC):
    @abstractmethod
    def describe_image(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        """Return model output for one image and one prompt."""
