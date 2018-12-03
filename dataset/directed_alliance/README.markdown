# Directed Alliance Data


## Download

The data are stored in a single `csv` file `directed_alliance_data.csv` (46Mb),
which can be downloaded by

    python get_data.py

Since the data file is arguably large, it is not recommended to check it in to the
GitHub repo.

The `SHA-256` and `MD5` checksums are as follows:

    SHA-256: ff1bacd5b0809478a25eb4775977d1bb176fbd669ecc12c0314e8dd9328ff7b7
    MD5:     6a8c112b75bc5e39b70998c62a1bbbca

## Content

- `country_a & country_b`: the pair of countries that can potentially form alliances (edges in network parlance)

- `year`: this is the time indicator for each year (data spans from 1948 to 2009

- `polity2_a`: this is the level of democracy (a numerical indicator from -10 to 10) where higher values indicate that a country_a is more democratic

- `polity2_b`: this is the level of democracy (a numerical indicator from -10 to 10) where higher values indicate that a country_b is more democratic

- `defense`: a binary indicator if country_a and country_b were in a defensive alliance with each other in year t

- `countrya_agereg`: this is a count meter that indicates the duration that countrya first experienced a regime change (structural shock)

- `countrya_agedem`: this is count meter for when the country first transitioned into a democracy

## Possible Directions

A hypothesis would be:
*the timing of countrya_agedem or countrya_agereg affects the probability of forming defense alliances with other states (the search for new friends).*
