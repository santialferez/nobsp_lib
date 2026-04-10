"""Public wrappers around the research NObSP package."""

from .models import SmallConvNet
from .plotting import (
    overlay_heatmap,
    plot_importance_heatmap,
    plot_tabular_feature_curves,
    plot_training_history,
    plot_vision_gallery,
    plot_vision_split_gallery,
    select_vision_heatmap,
)
from .tabular import EasyTabularExplainer, TabularExplainer
from .vision import EasyVisionExplainer, VisionExplainer

__all__ = [
    "TabularExplainer",
    "VisionExplainer",
    "EasyTabularExplainer",
    "EasyVisionExplainer",
    "SmallConvNet",
    "overlay_heatmap",
    "plot_importance_heatmap",
    "plot_tabular_feature_curves",
    "plot_training_history",
    "plot_vision_gallery",
    "plot_vision_split_gallery",
    "select_vision_heatmap",
]
