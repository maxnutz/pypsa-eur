#!/bin/bash

export GRB_LICENSE_FILE=/home/fs71375/joph_vsc/gurobi.lic
snakemake --slurm --jobs 8000 --use-conda --rerun-incomplete


