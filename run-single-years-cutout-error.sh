#!/bin/bash


for year in {2017..2023}; do
echo $year

chmod 700 cutouts/europe-$year-era5.nc
snakemake -call prepare_elec_networks --touch --configfile ../test-configs/single-years/config.electricity.europe.$year.yaml 
snakemake -call prepare_elec_networks --configfile ../test-configs/single-years/config.electricity.europe.$year.yaml 

done

echo All done
