import pypsa
import pandas as pd
import os.path
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from os import listdir
from os.path import isfile, join

opt_name = {"Store": "e", "Line": "s", "Transformer": "s"}

def calculate_costs(n):
    
    final_capital = pd.DataFrame()
    final_marginal = pd.DataFrame()

    for c in n.iterate_components(n.branch_components | n.controllable_one_port_components ^ {"Load"}):
        
        capital_costs = c.df.capital_cost * c.df[opt_name.get(c.name, "p") + "_nom_opt"]
        capital_costs_grouped = capital_costs.groupby(c.df.carrier).sum()

        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=["capital"])
        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=[c.list_name])
        
        
        final_capital = pd.concat([final_capital, capital_costs_grouped])
        #costs = costs.reindex(capital_costs_grouped.index.union(costs.index))

        #costs.loc[capital_costs_grouped.index, label] = capital_costs_grouped

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings.generators, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.0] = 0.0
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0).sum()

        # correct sequestration cost
        if c.name == "Store":
            items = c.df.index[
                (c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.0)
            ]
            c.df.loc[items, "marginal_cost"] = -20.0

        marginal_costs = p * c.df.marginal_cost

        marginal_costs_grouped = marginal_costs.groupby(c.df.carrier).sum()

        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=["marginal"])
        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=[c.list_name])
        
        final_marginal = pd.concat([final_marginal, marginal_costs_grouped])
        

    return(final_capital, final_marginal)

        #costs = costs.reindex(marginal_costs_grouped.index.union(costs.index))

        #costs.loc[marginal_costs_grouped.index, label] = marginal_costs_grouped

    # add back in all hydro
    # costs.loc[("storage_units", "capital", "hydro"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="hydro", "p_nom"].sum()
    # costs.loc[("storage_units", "capital", "PHS"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="PHS", "p_nom"].sum()
    # costs.loc[("generators", "capital", "ror"),label] = (0.02)*3e6*n.generators.loc[n.generators.group=="ror", "p_nom"].sum()

def demand(n, p):
    return n.loads_t.p_set.mul(n.snapshot_weightings["generators"], axis=0).sum().sum() * 1e-6

def objective(n, p):
    return n.objective

def power_capacity(n, carrier):
    g = n.generators
    return g.loc[g['carrier'].str.contains(carrier)]['p_nom_opt'].sum() * 1e-6

def power_generation(n, carrier):
    g = n.generators_t.p.mul(n.snapshot_weightings["generators"], axis=0)
    g_sum = g.sum()
    return g_sum[g_sum.index.str.contains(carrier)].sum() * 1e-9

def avg_power_price(n, p):
    intersect = n.buses_t.marginal_price.columns.intersection(n.buses[n.buses.carrier == "AC"].index)
    return n.buses_t.marginal_price[intersect].mean(axis=1).mean()
  
def max_power_price(n, p):
    intersect = n.buses_t.marginal_price.columns.intersection(n.buses[n.buses.carrier == "AC"].index)
    return n.buses_t.marginal_price[intersect].mean(axis=1).max()

def all_power_prices(n, p):
    intersect = n.buses_t.marginal_price.columns.intersection(n.buses[n.buses.carrier == "AC"].index)
    return n.buses_t.marginal_price[intersect].mean(axis=1)

def all_non_power_prices(n, p):
    intersect = n.buses_t.marginal_price.columns.intersection(n.buses[n.buses.carrier != "AC"].index)
    return n.buses_t.marginal_price[intersect].mean(axis=1)

def all_power_prices_country(n, c):
    intersect = n.buses_t.marginal_price.columns.intersection(n.buses[(n.buses.carrier == "AC") & (n.buses.country == c)].index)
    return n.buses_t.marginal_price[intersect].mean(axis=1)

def extract_time_series(f, p=""):
    dat = np.array([f(networks[key], p) for key in networks.keys()])
    return(dat)

def extract_summary_info_cost():
    l = [[key, calculate_costs(networks[key])] for key in networks.keys()] 
    columns = [l[i][0] for i in range(0, len(l))]
    
    capital = [l[i][1][0] for i in range(0, len(l))]
    df_capital = pd.concat(capital, axis=1)
    df_capital.columns = columns
    
    marginal = [l[i][1][1] for i in range(0, len(l))]
    df_marginal = pd.concat(marginal, axis=1)
    df_marginal.columns = columns    
    return (df_capital, df_marginal)

def plot_data_from_networks(f, second_param=""):
    dat = np.array([[key, f(networks[key], second_param)] for key in networks.keys()])
    df = pd.DataFrame(dat)
    df.shape
    df.columns = ["Year", "Value"]
    df = df.sort_values(by=["Value"])
    df["Year_str"] = df["Year"].astype(str)
    plt.figure(figsize=(20, 10))
    plt.bar(df["Year_str"], df["Value"])
    plt.xticks(rotation=90)
    
    
def load_original_networks(start, end):
    networks = {}
    for year in range(start, end + 1):
        file = f'/data/users/jschmidt/pypsa-eur-download-climate/pypsa-eur/results/weather_year_{year}/postnetworks/elec_s_37_lv1.5__Co2L0-3H-T-H-B-I-A-dist1_2050.nc'
        if os.path.exists(file):
            print(year)
            n = pypsa.Network(file)
            networks.update({year: n}) 
            
    return networks
    
def operate_year_with_different_weather_year(year_system, year_weather):
    
    file = f'/data/users/jschmidt/pypsa-eur-download-climate/pypsa-eur/results/weather_year_{year_system}/postnetworks/elec_s_37_lv1.5__Co2L0-3H-T-H-B-I-A-dist1_2050.nc'
    n_new = pypsa.Network(file)
    file = f'/data/users/jschmidt/pypsa-eur-download-climate/pypsa-eur/results/weather_year_{year_weather}/postnetworks/elec_s_37_lv1.5__Co2L0-3H-T-H-B-I-A-dist1_2050.nc'
    n_weather_year = pypsa.Network(file)
        
    n_new.generators["p_nom"] = n_new.generators["p_nom_opt"]
    n_new.generators["p_nom_extendable"] = False
    n_new.lines["s_nom"] = n_new.lines["s_nom_opt"]
    n_new.lines["s_nom_extendable"] = False
    n_new.storage_units["p_nom"] = n_new.storage_units["p_nom_opt"]
    n_new.storage_units["p_nom_extendable"] = False
    n_new.stores["e_nom"] = n_new.stores["e_nom_opt"]
    n_new.stores["e_nom_extendable"] = False
    n_new.generators_t.p_max_pu[:] = n_weather_year.generators_t.p_max_pu.values
    
    kwargs = {"solver_name": "gurobi", 
          "threads": 4, 
          "method": 2,
         "crossover": 0,
         "BarConvTol": 1.e-3,
         "Seed": 123,
         "AggFill": 0,
         "PreDual": 0,
         "GURO_PAR_BARDENSETHRESH": 200}
    n_new.optimize(**kwargs)
    
    dir = f'/data/users/jschmidt/pypsa-eur-download-climate/pypsa-eur/results/weather_year_{year_system}/postnetworks/operational/'
    Path(dir).mkdir(parents=True, exist_ok=True)
    n_new.export_to_netcdf(f'{dir}{year_weather}.nc')

def operate_year_with_adjusted_renewables(year_system, adjust_factor):
    n_new = networks[year_system].copy()
    n_new.generators["p_nom"] = n_new.generators["p_nom_opt"]
    n_new.generators["p_nom_extendable"] = False
    n_new.links["p_nom"] = n_new.links["p_nom_opt"]
    n_new.links["p_nom_extendable"] = False
    n_new.lines["s_nom"] = n_new.lines["s_nom_opt"]
    n_new.lines["s_nom_extendable"] = False
    n_new.storage_units["p_nom"] = n_new.storage_units["p_nom_opt"]
    n_new.storage_units["p_nom_extendable"] = False
    n_new.stores["e_nom"] = n_new.stores["e_nom_opt"]
    n_new.stores["e_nom_extendable"] = False
    generators_to_set_0 = n_new.generators[n_new.generators["carrier"].isin(["onwind", "ror", "offwind-ac", "offwind-dc", "solar"])].index
    n_new.generators_t.p_max_pu[generators_to_set_0] = n_new.generators_t.p_max_pu[generators_to_set_0] * adjust_factor
    for bus in n_new.buses[n_new.buses.carrier == "AC"].index:
        n_new.add("Generator", f'{bus} VOLL',
            bus=bus,
            carrier="VOLL",
            marginal_cost=2000,a
            capital_cost=0.1,
            p_nom=1000000,
            p_nom_extendable=False,
            control="")
    
    kwargs = {"solver_name": "gurobi", 
          "threads": 4, 
          "method": 2,
         "crossover": 0,
         "BarConvTol": 1.e-3,
         "Seed": 123,
         "AggFill": 0,
         "PreDual": 0,
         "GURO_PAR_BARDENSETHRESH": 200}
    n_new.optimize(**kwargs)
    return n_new


def rerun():
    networks = load_original_networks()
    n_1941_1942 = operate_year_with_weather_year(1941, 1942)
    networks[19411942] = n_1941_1942
    n_1942_1941 = operate_year_with_weather_year(1942, 1941)
    networks[19421941] = n_1942_1941
    n_1941_0 = operate_year_without_renewables(1941)
    networks[19410] = n_1941_0
    n_1942_0 = operate_year_without_renewables(1942)
    networks[19420] = n_1942_0
    return networks

def export_all_networks():
    [networks[key].export_to_netcdf(f'/data/users/jschmidt/pypsa-eur-download-climate/pypsa-eur/results/operational-runs/{key}.ncdf') for key in networks.keys()]
    
def load_all_networks():
    networks = {}
    mypath = '/data/users/jschmidt/pypsa-eur-download-climate/pypsa-eur/results/operational-runs/'
    for f in listdir(mypath):
        n = pypsa.Network()
        print(join(mypath, f))
        n.import_from_netcdf(join(mypath, f))
        networks.update({int(os.path.splitext(f)[0]) : n})  
    return networks

def main():
    operate_year_with_different_weather_year(2001, 2007)

if __name__ == "__main__":
    main()

