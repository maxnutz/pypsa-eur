YEAR=1974
echo $YEAR
chmod 700 cutouts/europe-$YEAR-era5.nc

snakemake -call prepare_elec_networks --touch --configfile ../test-configs/single-years/config.electricity.europe.$YEAR.yaml
snakemake -call prepare_elec_networks --configfile ../test-configs/single-years/config.electricity.europe.$YEAR.yaml

