#!/bin/bash


for year in {2002..2023}; do
echo $year

snakemake -call prepare_elec_networks --configfile ../test-configs/single-years/config.electricity.europe.$year.yaml 

done

#do	
#done

echo All done
