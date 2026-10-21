"""Count preparation and training-fitted classical transforms."""
from .classical import prepare, fit_preprocessing, transform
# Bootstrap is imported lazily to avoid initialization cycles with shared utilities.
def download_data(*args, **kwargs):
    from .bootstrap import download_data as run
    return run(*args, **kwargs)
def preprocess_data(**kwargs):
    from .bootstrap import preprocess_data as run
    return run(**kwargs)
