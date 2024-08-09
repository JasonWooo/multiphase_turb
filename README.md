# Multiphase Turbulence Project

Zewei "Jason" Wu (1), with Hitesh Kishore Das (2)

  (1) Dept. of Astronomy & Astrophysics, The University of Chicago
  
  (2) Max Planck Institute for Astrophysics
  
Summer 2024

This is the collection of codes, scripts, figures, and data for the multiphase turbulence project.

The version on github is this [repo](https://github.com/JasonWooo/multiphase_turb), which does not contain `data`, `saves`, and a few other directories.


# Directory informations

## athena

Hitesh's code on `Athena++`, up to date with the forked github [repo](https://github.com/HiteshKishoreDas/athena_fork_turb_box/tree/3-phase)

## athena_pp

The original `Athena++` code

## codes

Jason's codes for the project:

* `funcs`
  * basic functions & units
* `jason`
  * basic functions & units
* `plotting/`
  * plotting codes, including the `paper.py` used for making paper figures
* `processing/`
  * the python code to process and save variables into `save/`
* `trialgen/`
  * code to generate standalone trials or new trials from old ones
  * run `trial_lookup.sh` (`trial_lookup.py`) to generate a csv containing all information about the files (`out/trial_params.csv`)

## notebooks

* The collection of notebooks I've used throughout the project
* Most of these were before I serialized everything in `code`
* `fig` subfolder has a bunch of gifs and html files generated early in the research

## data

* Contains all data for the project. Labeled by `date_mach_rcl/lshat` folders
  * `cloud/` subfolder contains all cloud related files
  * `turb/` subfolder (if not empty) contains all turbulence-driving related files, since runs with the same mach number cross reference turbulent boxes
* `athena_turb` is the `Athena++` executable invoked in all .sh files, compiled from `athena`

## saves

* Data products saved from scripts
* `cloud_8e2_new.csv` contains all the reliable runs we ended up analyzing for the paper. The others not included here might have issues with pressure floor, radius scales, etc.
* `trial_params.csv` contains information on all the runs under `data/`. This file is generated from `trialgen/trial_lookup.sh`

## figures

* All the paper figures, in pdf format
* `animations` includes all the animations and their corresponding jpeg files by frame
* Generated from functions in `codes/plotting/paper.py`. The functional calls are in `notebooks/paper_figures.ipynb`

## hitesh_package

* Hitesh's own package on the github [repo](https://github.com/HiteshKishoreDas/own_package/tree/main)

## out

* The output (`.out`) & error (`.err`) files labeled by date

## scripts

* Shell scripts (`.sh`) used in the early days of the project