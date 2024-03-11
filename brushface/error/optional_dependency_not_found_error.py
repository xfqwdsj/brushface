from typing import Optional


class OptionalDependencyNotFoundError(Exception):
    dependency_name: str
    extra: Optional[str]

    def __init__(self, dependency_name: str, extra: Optional[str] = None):
        self.dependency_name = dependency_name
        self.extra = extra

        message = f"Optional dependency `{dependency_name}` not found."
        if extra:
            message += f" You can install it by adding extra `{extra}`."
        super().__init__(message)
