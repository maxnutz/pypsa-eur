# Open issue

## open snakemake steps
3987 open steps


## snakemake crashes due to netcdf error - has to be checked in detail.

## cutout.runoff breaks for certain or all years
added print statements at line 833 for debugging reasons
/home/jschmidt/.conda/envs/pypsa-eur/lib/python3.11/site-packages/atlite/convert.py
 
## clean and make branch with config
should clean repo and make a new branch.

## Important changes made for multiple weather years
- scripts/build_electricity_demand.py: added new parameter supplement_synthetic_always and included how it worked
- scripts/build_hydro_profile.py: fixed code at start which updates eia data
- scripts/make_summary.py: skip calculate market value

## Issues operational vs. system optimization runs
- when running 1941 system with 1942 data, the cost is 345 billion
- when running 1941 system with all timeseries set to 0, the cost is ~650 billion and it solves, very strange, has to be assessed
- when running 1941 system with 1941 data without expansion, the cost is XX billion


