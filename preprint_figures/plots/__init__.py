"""Plotting layer for the preprint figures.

Each module renders one analysis group's figures from the analysis cache. The
central label/title/legend registry lives in ``figures_spec.py``; the render
functions here pull their text from the passed ``FigureSpec`` so the SAME
function draws a standalone single-axis figure and a ``subplot_mosaic`` cell at
identical, locked fontsizes.
"""
