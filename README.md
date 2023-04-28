# WC-Density-Drag-Vegetation

In this project, we are making changes to the Water Column model of CE200B to accommodate a variable drag coefficient. The purpose of this variable drag coefficient is to describe how changes in vegetation density in the water column affect the velocities. The data on vegetation density comes from field studies carried out by the USGS in the South Bay. This report investigates how various scaling coefficients influence the impact of vegetation density on the Water Column model.
## How To Use

The water column model is found under the water_column_model/ folder, with the Column object being described in column.py and the time-stepping algorithm in advance.py

The wc_modeling.ipynb notebook calls the water column model for the given site parameters and requires 2 inputs: The number of different drag scale runs needed and an array with the different values of alpha we want to work with.

The wc_figures.ipynb notebook is used for visualizations of model runs.
