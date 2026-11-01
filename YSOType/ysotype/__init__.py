"""YSOType: spectral types, extinction and veiling of young stellar objects."""
__version__ = "1.0.0"

from .spectrum import Spectrum, read_spectrum          # noqa: F401
from .templates import TemplateGrid                    # noqa: F401
from .fit import FitConfig, Fitter, FitResult, fit_spectrum  # noqa: F401
from .spt import TeffScale, spt_to_code, code_to_spt   # noqa: F401
