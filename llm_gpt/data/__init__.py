"""
data/__init__.py - Módulo de dados
"""

from .dataset import (
    TextDataset,
    InMemoryTextDataset,
    create_dataloaders,
    prepare_dataset,
    generate_dummy_data
)

__all__ = [
    'TextDataset',
    'InMemoryTextDataset',
    'create_dataloaders',
    'prepare_dataset',
    'generate_dummy_data'
]
