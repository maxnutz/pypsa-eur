# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Create summary CSV files for all scenario runs including costs, capacities,
capacity factors, curtailment, energy balances, prices and other metrics.
"""

import logging

import pandas as pd
import pypsa
from _helpers import configure_logging, set_scenario_config

idx = pd.IndexSlice
logger = logging.getLogger(__name__)

OUTPUTS = [
    "costs",
    "capacities",
    "energy",
    "energy_balance",
    "capacity_factors",
    "metrics",
    "curtailment",
    "prices",
    "weighted_prices",
    "market_values",
    "nodal_costs",
    "nodal_capacities",
    "nodal_energy_balance",
    "nodal_capacity_factors",
]


def assign_carriers(n: pypsa.Network) -> None:
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"


def assign_locations(n: pypsa.Network) -> None:
    for c in n.iterate_components(n.one_port_components):
        c.df["location"] = c.df.bus.map(n.buses.location)

    for c in n.iterate_components(n.branch_components):
        c_bus_cols = c.df.filter(regex="^bus")
        locs = c_bus_cols.apply(lambda c: c.map(n.buses.location)).sort_index(axis=1)
        # Use first location that is not "EU"; take "EU" if nothing else available
        c.df["location"] = locs.apply(
            lambda row: next(
                (loc for loc in row.dropna() if loc != "EU"),
                "EU",
            ),
            axis=1,
        )


def calculate_nodal_capacity_factors(n: pypsa.Network) -> pd.Series:
    """
    Calculate the regional dispatched capacity factors / utilisation rates for each technology carrier based on location bus attribute.
    """
    comps = n.one_port_components ^ {"Store"} | n.passive_branch_components
    return n.statistics.capacity_factor(comps=comps, groupby=["location", "carrier"])


def calculate_capacity_factors(n: pypsa.Network) -> pd.Series:
    """
    Calculate the average dispatched capacity factors / utilisation rates for each technology carrier.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["component", "carrier"]
    """

    comps = n.one_port_components ^ {"Store"} | n.passive_branch_components
    return n.statistics.capacity_factor(comps=comps).sort_index()


def calculate_nodal_costs(n: pypsa.Network) -> pd.Series:
    """
    Calculate optimized regional costs for each technology split by marginal and capital costs and based on location bus attribute.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["cost", "component", "location", "carrier"]
    """
    grouper = ["location", "carrier"]
    costs = pd.concat(
        {
            "capital": n.statistics.capex(groupby=grouper),
            "marginal": n.statistics.opex(groupby=grouper),
        }
    )
    costs.index.names = ["cost", "component", "location", "carrier"]

    return costs


def calculate_costs(n: pypsa.Network) -> pd.Series:
    """
    Calculate optimized total costs for each technology split by marginal and capital costs.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["cost", "component", "carrier"]
    """
    costs = pd.concat(
        {
            "capital": n.statistics.capex(),
            "marginal": n.statistics.opex(),
        }
    )
    costs.index.names = ["cost", "component", "carrier"]

    return costs


def calculate_nodal_capacities(n: pypsa.Network) -> pd.Series:
    """
    Calculate optimized regional capacities for each technology relative to bus/bus0 based on location bus attribute.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["component", "location", "carrier"]
    """
    return n.statistics.optimal_capacity(groupby=["location", "carrier"])


def calculate_capacities(n: pypsa.Network) -> pd.Series:
    """
    Calculate optimized total capacities for each technology relative to bus/bus0.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["component", "carrier"]
    """
    return n.statistics.optimal_capacity()


def calculate_curtailment(n: pypsa.Network) -> pd.Series:
    """
    Calculate the curtailment of electricity generation technologies in percent.
    """

    carriers = ["AC", "low voltage"]

    duration = n.snapshot_weightings.generators.sum()

    curtailed_abs = n.statistics.curtailment(
        bus_carrier=carriers, aggregate_across_components=True
    )
    available = (
        n.statistics.optimal_capacity("Generator", bus_carrier=carriers) * duration
    )

    curtailed_rel = curtailed_abs / available * 100

    return curtailed_rel.sort_index()


def calculate_energy(n: pypsa.Network) -> pd.Series:
    """
    Calculate the net energy supply (positive) and consumption (negative) by technology carrier across all ports.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["component", "carrier"]
    """
    return n.statistics.energy_balance(groupby="carrier").sort_values(ascending=False)


def calculate_energy_balance(n: pypsa.Network) -> pd.Series:
    """
    Calculate the energy supply (positive) and consumption (negative) by technology carrier for each bus carrier.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["component", "carrier", "bus_carrier"]

    Examples
    --------
    >>> eb = calculate_energy_balance(n)
    >>> eb.xs("methanol", level='bus_carrier')
    """
    return n.statistics.energy_balance().sort_values(ascending=False)


def calculate_nodal_energy_balance(n: pypsa.Network) -> pd.Series:
    """
    Calculate the regional energy balances (positive values for supply, negative values for consumption) for each technology carrier and bus carrier based on the location bus attribute.

    Returns
    -------
    pd.Series
        MultiIndex Series with levels ["component", "carrier", "location", "bus_carrier"]

    Examples
    --------
    >>> eb = calculate_nodal_energy_balance(n)
    >>> eb.xs(("AC", "BE0 0"), level=["bus_carrier", "location"])
    """
    return n.statistics.energy_balance(groupby=["carrier", "location", "bus_carrier"])


def calculate_metrics(n: pypsa.Network) -> pd.Series:
    """
    Calculate system-level metrics, e.g. shadow prices, grid expansion, total costs.
    Also calculate average, standard deviation and share of zero hours for electricity prices.
    """

    metrics = {}

    dc_links = n.links.query("carrier == 'DC'")
    metrics["line_volume_DC"] = dc_links.eval("length * p_nom_opt").sum()
    metrics["line_volume_AC"] = n.lines.eval("length * s_nom_opt").sum()
    metrics["line_volume"] = metrics["line_volume_AC"] + metrics["line_volume_DC"]

    metrics["total costs"] = n.statistics.capex().sum() + n.statistics.opex().sum()

    buses_i = n.buses.query("carrier == 'AC'").index
    prices = n.buses_t.marginal_price[buses_i]

    # threshold higher than marginal_cost of VRE
    zero_hours = prices.where(prices < 0.1).count().sum()
    metrics["electricity_price_zero_hours"] = zero_hours / prices.size
    metrics["electricity_price_mean"] = prices.unstack().mean()
    metrics["electricity_price_std"] = prices.unstack().std()

    if "lv_limit" in n.global_constraints.index:
        metrics["line_volume_limit"] = n.global_constraints.at["lv_limit", "constant"]
        metrics["line_volume_shadow"] = n.global_constraints.at["lv_limit", "mu"]

    if "CO2Limit" in n.global_constraints.index:
        metrics["co2_shadow"] = n.global_constraints.at["CO2Limit", "mu"]

    if "co2_sequestration_limit" in n.global_constraints.index:
        metrics["co2_storage_shadow"] = n.global_constraints.at[
            "co2_sequestration_limit", "mu"
        ]

    return pd.Series(metrics).sort_index()


def calculate_prices(n: pypsa.Network) -> pd.Series:
    """
    Calculate time-averaged prices per carrier.
    """
    return n.buses_t.marginal_price.mean().groupby(n.buses.carrier).mean().sort_index()


def calculate_weighted_prices(n: pypsa.Network) -> pd.Series:
    """
    Calculate load-weighted prices per bus carrier.
    """
    carriers = n.buses.carrier.unique()

    weighted_prices = {}

    for carrier in carriers:
        load = n.statistics.withdrawal(
            groupby="bus",
            aggregate_time=False,
            bus_carrier=carrier,
            aggregate_across_components=True,
        ).T

        if not load.empty and load.sum().sum() > 0:
            price = n.buses_t.marginal_price.loc[:, n.buses.carrier == carrier]
            price = price.reindex(columns=load.columns, fill_value=1)

            weights = n.snapshot_weightings.generators
            a = weights @ (load * price).sum(axis=1)
            b = weights @ load.sum(axis=1)
            weighted_prices[carrier] = a / b

    return pd.Series(weighted_prices).sort_index()


def calculate_market_values(n: pypsa.Network) -> pd.Series:
    """
    Calculate market values for electricity.
    """
    return (
        n.statistics.market_value(bus_carrier="AC", aggregate_across_components=True)
        .sort_values()
        .dropna()
    )

<<<<<<< HEAD
=======
    link_loads = {
        "electricity": [
            "heat pump",
            "resistive heater",
            "battery charger",
            "H2 Electrolysis",
        ],
        "heat": ["water tanks charger"],
        "urban heat": ["water tanks charger"],
        "space heat": [],
        "space urban heat": [],
        "gas": ["OCGT", "gas boiler", "CHP electric", "CHP heat"],
        "H2": ["Sabatier", "H2 Fuel Cell"],
    }

    for carrier, value in link_loads.items():
        if carrier == "electricity":
            suffix = ""
        elif carrier[:5] == "space":
            suffix = carrier[5:]
        else:
            suffix = " " + carrier

        buses = n.buses.index[n.buses.index.str[2:] == suffix]

        if buses.empty:
            continue

        if carrier in ["H2", "gas"]:
            load = pd.DataFrame(index=n.snapshots, columns=buses, data=0.0)
        else:
            load = n.loads_t.p_set[buses.intersection(n.loads.index)]

        for tech in value:
            names = n.links.index[n.links.index.to_series().str[-len(tech) :] == tech]

            if not names.empty:
                load += (
                    n.links_t.p0[names].T.groupby(n.links.loc[names, "bus0"]).sum().T
                )

        # Add H2 Store when charging
        # if carrier == "H2":
        #    stores = n.stores_t.p[buses+ " Store"].groupby(n.stores.loc[buses+ " Store", "bus"],axis=1).sum(axis=1)
        #    stores[stores > 0.] = 0.
        #    load += -stores

        weighted_prices.loc[carrier, label] = (
            load * n.buses_t.marginal_price[buses]
        ).sum().sum() / load.sum().sum()

        # still have no idea what this is for, only for debug reasons.
        if carrier[:5] == "space":
            logger.debug(load * n.buses_t.marginal_price[buses])

    return weighted_prices


def calculate_market_values(n, label, market_values):
    # Warning: doesn't include storage units
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8]
        }, index=['w', 'x', 'y', 'z'])
    
    return df
    carrier = "AC"

    buses = n.buses.index[n.buses.carrier == carrier]

    ## First do market value of generators ##
    
    #generators = n.generators.index[n.buses.loc[n.generators.bus, "carrier"] == carrier]
    generators = n.generators.index    
    techs = n.generators.loc[generators, "carrier"].value_counts().index
    


    market_values = market_values.reindex(market_values.index.union(techs))

    for tech in techs:
        gens = generators[n.generators.loc[gen_2014.objectivenerators, "carrier"] == tech]

        dispatch = (
            n.generators_t.p[gens]
            .T.groupby(n.generators.loc[gens, "bus"])
            .sum()
            .T.reindex(columns=buses, fill_value=0.0)
        )
        revenue = dispatch * n.buses_t.marginal_price[buses]

        if total_dispatch := dispatch.sum().sum():
            market_values.at[tech, label] = revenue.sum().sum() / total_dispatch
        else:
            market_values.at[tech, label] = np.nan

    ## Now do market value of links ##

    for i in ["0", "1"]:
        all_links = n.links.index[n.buses.loc[n.links["bus" + i], "carrier"] == carrier]

        techs = n.links.loc[all_links, "carrier"].value_counts().index

        market_values = market_values.reindex(market_values.index.union(techs))

        for tech in techs:
            links = all_links[n.links.loc[all_links, "carrier"] == tech]

            dispatch = (
                n.links_t["p" + i][links]
                .T.groupby(n.links.loc[links, "bus" + i])
                .sum()
                .T.reindex(columns=buses, fill_value=0.0)
            )

            revenue = dispatch * n.buses_t.marginal_price[buses]

            if total_dispatch := dispatch.sum().sum():
                market_values.at[tech, label] = revenue.sum().sum() / total_dispatch
            else:
                market_values.at[tech, label] = np.nan

    return market_values


def calculate_price_statistics(n, label, price_statistics):
    price_statistics = price_statistics.reindex(
        price_statistics.index.union(
            pd.Index(["zero_hours", "mean", "standard_deviation"])
        )
    )

    buses = n.buses.index[n.buses.carrier == "AC"]

    threshold = 0.1  # higher than phoney marginal_cost of wind/solar

    df = pd.DataFrame(data=0.0, columns=buses, index=n.snapshots)

    df[n.buses_t.marginal_price[buses] < threshold] = 1.0

    price_statistics.at["zero_hours", label] = df.sum().sum() / (
        df.shape[0] * df.shape[1]
    )

    price_statistics.at["mean", label] = (
        n.buses_t.marginal_price[buses].unstack().mean()
    )

    price_statistics.at["standard_deviation", label] = (
        n.buses_t.marginal_price[buses].unstack().std()
    )

    return price_statistics


def make_summaries(networks_dict):
    outputs = [
        "nodal_costs",
        "nodal_capacities",
        "nodal_cfs",
        "cfs",
        "costs",
        "capacities",
        "curtailment",
        "energy",
        "supply",
        "supply_energy",
        "prices",
        "weighted_prices",
        "price_statistics",
        "market_values",
        "metrics",
    ]

    columns = pd.MultiIndex.from_tuples(
        networks_dict.keys(),
        names=["cluster", "ll", "opt", "planning_horizon"],
    )

    df = {output: pd.DataFrame(columns=columns, dtype=float) for output in outputs}
    for label, filename in networks_dict.items():
        logger.info(f"Make summary for scenario {label}, using {filename}")

        n = pypsa.Network(filename)

        assign_carriers(n)
        assign_locations(n)

        for output in outputs:
            df[output] = globals()["calculate_" + output](n, label, df[output])

    return df


def to_csv(df):
    for key in df:
        df[key].to_csv(snakemake.output[key])

>>>>>>> pypsa-eur_joph/fix-hydropower-and-load-bugs

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "make_summary",
            clusters="5",
            opts="",
            sector_opts="",
            planning_horizons="2030",
            configfiles="config/test/config.overnight.yaml",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)
    assign_carriers(n)
    assign_locations(n)

    n.statistics.set_parameters(nice_names=False, drop_zero=False)

    for output in OUTPUTS:
        globals()["calculate_" + output](n).to_csv(snakemake.output[output])
