"""Unitree G1 assets for psm.

Robot constants live in ``g1_constants.py`` and ``g1_23dof_constants.py``. This
``__init__`` does not re-export them so imports avoid re-entry when mjlab loads
the ``mjlab.tasks`` entry point during constant modules' ``mjlab`` imports.
Use ``psm.assets.unitree_g1.g1_constants`` or re-exports in
``psm.assets``.
"""
