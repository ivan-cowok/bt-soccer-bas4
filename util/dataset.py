# Global imports
from collections import Counter

# Local imports
from util.io import load_text
from util.constants import F3SET_ELEMENTS


def load_classes(file_name, active_class_names=None):
    """
    Map class name -> label index (1..N). Index 0 is reserved for background.

    If active_class_names is a non-empty list, only those names get indices 1..K
    in list order; any label not in this map is treated as background (same as
    skipping unknown labels in the dataset loaders). Names must appear in class.txt.

    If active_class_names is None or empty, every line in class.txt gets an index
    (original behavior).
    """
    all_names = load_text(file_name)
    if not all_names:
        raise ValueError(f'No classes in {file_name}')
    full_set = set(all_names)
    if not active_class_names:
        return {x: i + 1 for i, x in enumerate(all_names)}
    if not isinstance(active_class_names, (list, tuple)):
        raise TypeError('active_class_names must be a list or tuple of strings')
    missing = [n for n in active_class_names if n not in full_set]
    if missing:
        raise ValueError(
            f'active_class_names entries not found in {file_name}: {missing}. '
            f'Available names (first 20): {list(all_names)[:20]}'
        )
    dup = [n for n, c in Counter(active_class_names).items() if c > 1]
    if dup:
        raise ValueError(f'Duplicate names in active_class_names: {dup}')
    return {name: i + 1 for i, name in enumerate(active_class_names)}

def load_elements(file_name):
    elements = []
    elements_text = load_text(file_name)
    j = 0
    for category_length in F3SET_ELEMENTS:
        category_start = j
        category_end = j + category_length
        elements.append({elements_text[i]: i - category_start for i in range(category_start, category_end)})
        j += category_length
    
    return elements