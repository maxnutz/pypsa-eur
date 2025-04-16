# SPDX-FileCopyrightText: : 2023-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# This script helps to generate a scenarios.yaml file for PyPSA-Eur.
# You can modify the template to your needs and define all possible combinations of config values that should be considered.

if "snakemake" in globals():
    filename = snakemake.output[0]  # noqa: F821
else:
    filename = "../config/scenarios.yaml"

#import itertools

combined_scenarios = ""

for year in range(1941, 2024):
    template = f'''

weather_year_{year}:

    snapshots:
        start: "{year}-01-01"
        end: "{year + 1}-01-01"
        inclusive: "left"
  
    atlite:
        default_cutout: europe-era5-{year}
        cutouts:
        europe-era5-{year}:
            module: era5
            x: [-12., 42.]
            y: [33., 72]
            dx: 0.3
            dy: 0.3
            time: ['{year}', '{year}']
    
    renewable:
        onwind:
            cutout: europe-era5-{year}
        offwind-ac:
            cutout: europe-era5-{year}
        offwind-dc:
            cutout: europe-era5-{year}
        solar:
            cutout: europe-era5-{year}
        hydro:
            cutout: europe-era5-{year}
  
    solar_thermal:
            cutout: europe-era5-{year}
  
    sector:
            heat_demand_cutout: europe-era5-{year}

    lines:
        dynamic_line_rating:
            activate: true
            cutout: europe-era5-{year}


    '''
    combined_scenarios = combined_scenarios + template

with open(filename, "w") as f:
    f.write(combined_scenarios)


#config_values = dict(year=range(1941, 1943), year_1=range(1942, 1944))

#combinations = [
 #   dict(zip(config_values.keys(), values))
  #  for values in itertools.product(*config_values.values())
   # #for values in itertools.combinations(config_values.values(), 2)
    #]

#with open(filename, "w") as f:
 #   for i, config in enumerate(combinations):
  #      f.write(template.format(scenario_number=i, **config))
        
