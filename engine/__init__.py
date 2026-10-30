# -*- coding: utf-8 -*-
"""OCVR engine: trial plots x cleaned yield x multi-year NDVI zones x prices -> variable-rate vs flat-rate economics.

Modules (pure Python, no Streamlit):
    io          read vector / raster files, detect columns, field UTM
    prices      crop and input prices, product -> nutrient conversion, partial budget
    ndvi        multi-year NDVI layers: read GeoTIFFs, put them on one grid, normalise, temporal mean / CV
    ndvi_fetch  fetch peak-season Sentinel-2 NDVI per year from Copernicus (reuses GIT/NDVIMAP)
    zones       management zones from the NDVI stack or from an uploaded zone map; Rx polygons
    join        trial design, transition trims, yield polygons, as-applied, analysis units (plot x zone)
    response    yield response models (pooled, zone x rate, NDVI x rate, plateau)
    economics   optimum rates, VR vs flat (status quo and uniform optimum), bootstrap, cross-fit, sensitivity,
                realised trial economics
    treatments  strip-treatment comparison (VR vs flat vs none), RCBD ANOVA + Tukey
    report      PDF report and exports
    project     the project record (project.json) and the full pipeline
"""
__version__ = "1.1.0"
