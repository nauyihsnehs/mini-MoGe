from typing import Type

from .v2 import MoGeModel


def import_model_class_by_version(version: str) -> Type[MoGeModel]:
    if version != "v2":
        raise ValueError(f'Unsupported model version: {version}')
    return MoGeModel
