

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_renewable_profiles", clusters=38, technology="offwind-ac"
        )
        print("no snakemake found")
    print("i am here")
    #print(snakemake.params)
    params = snakemake.params.renewable["onwind"]
    print(params)
    print(params.get("min_p_max_pu", 0.0))
