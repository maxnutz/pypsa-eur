import pypsa
    
def operate_year_with_different_weather_year(year_system, year_weather, output):
    
    n_new = pypsa.Network(year_system)
    n_weather_year = pypsa.Network(year_weather)

    print(n_new)
    print(n_weather_year)
        
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
         #"BarConvTol": 0.1,
         "Seed": 123,
         "AggFill": 0,
         "PreDual": 0,
         "GURO_PAR_BARDENSETHRESH": 200}

    #kwargs = {"solver_name": "cplex",
    #        "threads": 4,
    #        "lpmethod": 4,
    #        "solutiontype": 2,
    #        "barrier.convergetol": 1.e-5,
    #        "feasopt.tolerance": 1.e-6}

    n_new.optimize(**kwargs)
    
    try:

        print(f'exporting {output}')
        n_new.export_to_netcdf(output)
        print("network exported!")
    except:
        assert(f'Problem exporting {output}')
         
    
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
            marginal_cost=2000,
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


#def main():
#    operate_year_with_different_weather_year("results/weather_year_2001/postnetworks/elec_s_37_lv1.5__Co2L0-3H-T-H-B-I-A-dist1_2050.nc",
 #                                            "results/weather_year_2001/postnetworks/elec_s_37_lv1.5__Co2L0-3H-T-H-B-I-A-dist1_2050.nc",
#                                             "results/weather_year_2001/postnetworks/operational/2001.nc") 

#if __name__ == "__main__":
#    main()
#else:
operate_year_with_different_weather_year(snakemake.input[0], snakemake.input[1], snakemake.output[0])
