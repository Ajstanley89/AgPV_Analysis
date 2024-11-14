#!/usr/bin/env python
# coding: utf-8

# # AgPV Index
# Exploratory data analysis for creation of an AG PV index: a composite scoring metric to identify USA counties with high cobenefits potential for agrivoltaics
# 
# * Data inputs: solar supply, weather hazards, energy burden, minority owned cropland

# ## Data Processing
# Things to do:
# 
# * ~aggregate NREL's solar supply data by county~
# * ~get relevant weather hazards data from csv file (hail and drought are positives, tornado is negative.)~
#     * might start ith overall agriculture burden for positives first
# * ~load energy burden data~
# * ~get minoirty owned data from R2R indices to start~
# * Estimate fraction of energy coops in county
#     * from Energy Information Administration. Need to combine several of provided datasets:
#     * Power generated/customers served by each utility, Counties each utility services, FIPS info
# * Suitable crops for ag pv
# * [True land footprint of solar energy](https://betterenergy.org/blog/the-true-land-footprint-of-solar-energy/)
# * 8-year slope values for each state's electricity (quads)
# * C-intensity of electricity for most recent year (2022) divided by the electricity (quads)
#  

# In[1]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import regex as re

from usda_qwikstats_utils import usda_qwikstats_utils
from qwi_utils import qwi_utils
from config import usda_qwik_stats_key, qwi_key
from requests import get
from typing import Callable, Union

data_dir = 'AgPV_data'
export_dir = 'Export_data'
graph_dir = 'AgPV_graphs'

usda_utils = usda_qwikstats_utils(usda_qwik_stats_key)

# vegetable commodities to get
veggie_commodities = ['beets', 'potatoes', 'radishes', 'tomatoes', 'peppers', 'carrots', 'spinach', 'lettuce']

usda_search_params = {'year':'2022',
                      'agg_level':'county',
                      'source_desc':'census',
                      'statisticcat_desc': 'Area Harvested',
                      'format':'JSON'
                     }

def get_usda_dfs(commodity: str, search_params:dict = usda_search_params, usda_utils:usda_qwikstats_utils = usda_utils, verbose: bool = False) -> pd.DataFrame:
    '''Calls usda qwikstats API. Returns pandas df for the commodity searched for'''
    print(f'Getting {commodity.title()} Data...')
    
    params_copy = search_params

    if verbose:
        print(f'Getting {commodity} data...')

    if 'commodity_desc' not in params_copy.keys():
        params_copy['commodity_desc'] = commodity
        
    # make api call
    r = usda_utils.get_qwikstats_data(params_copy)
    
    try:
        data = r.json()['data']
        if verbose:
            print(f'{commodity.title()} done!')
            
        return pd.DataFrame(data)
    
    except Exception as e:
        print(e)
        print(r.text)

def get_usda_economic_data(params: dict, suffix: str, domain_desc: str='TOTAL', fill_method: Union[Callable, int, float] = 0):
    """
    Gets economic data from qwik stats data base. Provides extra processing for filling na values compared to the get_usda_dfs function. 
    
    Why not just factor them into one function? Dunno. My code be janky like that.
    """
    try:
        # get data
        r = usda_utils.get_qwikstats_data(params)
        df = pd.DataFrame(r.json()['data'])

    except Exception as e:
        print(e)
        print(r)
        print(r.text)
        return None
    
    # clean data
    clean_df = clean_usda_values(df, suffix, domain_desc, fill_method=fill_method)
    clean_df['FIPS'] = clean_df['state_ansi'] + clean_df['county_ansi']
    
    return clean_df

def clean_usda_values(df: pd.DataFrame, suffix: str, domain_desc: str = 'TOTAL', fill_method: Union[Callable, int, float] = 0) -> pd.DataFrame:
    """
    Process the raw values returned from usda api results df. 
    
    -Filters for just county data
    -Creates specific column labeling data that was undisclosed
    -Removes ',' from numbers
    -Sets (D) and (Z) values to 0
    -Converts 'Value' column to numeric
    """
    if 'agg_level_desc' in df.columns:
        county_df = df.loc[df['agg_level_desc'] == 'COUNTY', :]
    elif 'Geo Level' in df.columns:
        county_df = df.loc[df['Geo Level'] == 'COUNTY', :]
    else:
        county_df = df.copy()
        
    if domain_desc is not None and 'domain_desc' in county_df.columns:
        county_df = county_df.loc[county_df['domain_desc'] == domain_desc]
        
    county_df.loc[:, [f'Undisclosed_{suffix}']] = county_df['Value'].apply(lambda x: x.strip().upper() == '(D)')
    county_df.loc[:, 'Value'] = county_df['Value'].apply(lambda x: x.strip().replace(',',''))
    county_df.loc[county_df['Value'] =='(Z)', 'Value'] = 0
    
    if isinstance(fill_method, Callable):
        county_df.loc[county_df['Value'] =='(D)', 'Value'] = fill_method(county_df.Value)
    else:
        county_df.loc[county_df['Value'] =='(D)', 'Value'] = fill_method
        
    county_df.loc[:, 'Value'] = county_df['Value'].astype(int, errors='ignore')
    
    return county_df

def get_qwi_dfs(qwi: qwi_utils, state: str) -> pd.DataFrame:
    """Queries the quarterly workforce indicator database to get county job counts for specific NAICS"""
    search_params = {'state': state,
                    'industry': ['111', '112'],
                    'time': '2022'}
    
    return qwi.make_qwi_df(**search_params)

if __name__ == "__main__":
    # # USDA Qwik stats
    # Crops like beets, tomatoes, peppers, and leafy greens seem to do well with agPV. e want to find counties with high amounts/percentages of crop area for these crops (beets, potatoes, radishes, tomatoes, peppers, carrots, spinach, lettuce, all berries, goats, and sheep)
    # 
    # May have to calculate livestock and crops seperately. I believe crops can be measured in area, but livestock might be measured in heads.
    print('Getting Veggies...')
    veggies_dfs = [get_usda_dfs(commodity, verbose=True) for commodity in veggie_commodities]

    veggies_df = pd.concat(veggies_dfs)
        
    veggies_clean_df = clean_usda_values(veggies_df, 'veggies')
    veggies_clean_df = veggies_clean_df.loc[(veggies_clean_df['unit_desc'] == 'ACRES') & \
                                            (veggies_clean_df['util_practice_desc'] == 'ALL UTILIZATION PRACTICES'), :]
    
    # save the unaggregated crops in case they need specific crop analyses
    veggies_clean_df.to_csv(os.path.join(data_dir, 'vegetable_acres.csv'))

    veggies_sum_df = veggies_clean_df.groupby(['state_ansi', 'county_ansi']).sum()[['Value', 'Undisclosed_veggies']]
    veggies_sum_df = veggies_sum_df.reset_index()
    veggies_sum_df['FIPS'] = veggies_sum_df['state_ansi'] + veggies_sum_df['county_ansi']
    veggies_sum_df = veggies_sum_df[['FIPS', 'Value', 'Undisclosed_veggies']]

    ### Get Berries Data
    print('Getting Berries...')
    berry_search_params = {'year':'2022',
                                'agg_level':'county',
                                'source_desc':'census',
                                'statisticcat_desc_LIKE': 'ACRES GROWN',
                                'format':'JSON'}

    berry_search_params['commodity_desc'] = 'Berry Totals'

    r = usda_utils.get_qwikstats_data(berry_search_params)

    berries_df = pd.DataFrame(r.json()['data'])

    berries_clean_df = clean_usda_values(berries_df, 'berries')

    # just keep 'acres' and 'all production practices'
    berries_clean_df = berries_clean_df.loc[(berries_clean_df['unit_desc'] == 'ACRES') & (berries_clean_df['prodn_practice_desc'] == 'ALL PRODUCTION PRACTICES'),:]

    berries_sum_df = berries_clean_df.groupby(['state_ansi', 'county_ansi']).sum()
    berries_sum_df = berries_sum_df.reset_index()
    berries_sum_df['FIPS'] = berries_sum_df['state_ansi'] + berries_sum_df['county_ansi']
    berries_sum_df = berries_sum_df[['FIPS', 'Value', 'Undisclosed_berries']]
    berries_sum_df.sort_values('Value', ascending=False)


    # ### Goats and Sheep
    # Goats and sheep don't have acres available. Need to use total sales instead
    print('Getting Livestock...')
    sheep_goats_path = '68E1129D-0FD5-3FC4-A812-EF6D3EBE6718.csv'

    sheep_goats_df = pd.read_csv(os.path.join(data_dir, sheep_goats_path), dtype={'State ANSI':str, 'County ANSI':str})

    sheep_goats_clean_df = clean_usda_values(sheep_goats_df, 'goats_sheep_totals')
    sheep_goats_clean_df['FIPS'] = sheep_goats_clean_df['State ANSI'] + sheep_goats_clean_df['County ANSI']
    sheep_goats_clean_df = sheep_goats_clean_df[['FIPS', 'Value', 'Undisclosed_goats_sheep_totals']]
    sheep_goats_clean_df = sheep_goats_clean_df.rename(columns={'Value':'Sheep_goats_total_sales'})

    # Demographics data from USDA
    sheep_goats_demo_df = pd.read_csv(os.path.join(data_dir, 'sheep_goats_minority.csv'), dtype={'State ANSI':str, 'County ANSI':str})
    sheep_goats_demo_df = clean_usda_values(sheep_goats_demo_df, 'goats_sheep_producers')
    sheep_goats_demo_df['FIPS'] = sheep_goats_demo_df['State ANSI'] + sheep_goats_demo_df['County ANSI']
    sheep_goats_demo_df = sheep_goats_demo_df[['FIPS', 'Year', 'Domain Category', 'Value', 'Undisclosed_goats_sheep_producers']]

    # Assume that minority ropland == all nonwhite producers
    white_value = 'PRODUCERS: (RACE = WHITE, ALONE OR COMBINED WITH OTHER RACES)'
    female_value = 'PRODUCERS: (GENDER = FEMALE)'

    sheep_goats_demo_df = sheep_goats_demo_df.loc[(sheep_goats_demo_df['Year'] == 2022)]

    sheep_goats_demo_df = sheep_goats_demo_df.loc[sheep_goats_demo_df['Domain Category'].isin([white_value, female_value]), ['FIPS', 'Domain Category', 'Value']]
    sheep_goats_demo_df = sheep_goats_demo_df.pivot_table(values='Value', index='FIPS', columns='Domain Category')
    sheep_goats_demo_df['Percent Nonwhite Sheep Goats Producers'] = 100 - sheep_goats_demo_df[white_value]
    sheep_goats_demo_df = sheep_goats_demo_df.rename(columns={female_value:'Percent Woman-owned Sheep Goats Producers'})
    sheep_goats_demo_df = sheep_goats_demo_df.drop(columns=white_value)
    sheep_goats_demo_df = sheep_goats_demo_df.fillna(0).reset_index()

    # berries and veggies can be added together since they're both in acres. Goats/sheep are in $
    crops_sum_df = veggies_sum_df.merge(berries_sum_df, on='FIPS', suffixes=('_veggies', '_berries'), how='outer')
    crops_sum_df['AgPV_crop_totals'] = crops_sum_df['Value_veggies'] + crops_sum_df['Value_berries']
    crops_sum_df

    # ### Solar Panel Land Use
    print('Getting Solar Panel Land Use...')
    # Land that isn't currently occupied by lots of solar panels. People might be more amenable to development if the landscape isn't already cluttered by existing solar panels
    solar_land_path = os.path.join(data_dir, 'solar_panel_land_use.csv')
    solar_land_df = pd.read_csv(solar_land_path, dtype={'FIPS':str})
    solar_land_df = solar_land_df[['FIPS', 'SolarPanelLandUse']]

    # ### Solar Supply Data NREL
    # 
    # Commonly cited NREL data for solar supply. I'm aggregating this by county.
    # 
    # https://www.nrel.gov/gis/solar-resource-maps.html
    print('Getting solar supply...')
    # get solar supply data
    solar_dir = 'solar-pv-reference-access-2023'
    solar_file = 'reference_access_2030_moderate_supply-curve.csv'

    solar_df = pd.read_csv(os.path.join(solar_dir, solar_file), dtype={'cnty_fips':str})
    solar_df['cnty_fips'] = solar_df['cnty_fips'].apply(lambda x: x.zfill(5))

    # get total for each county
    location_cols = ['cnty_fips', 'county', 'state']
    solar_supply_cols = ['capacity_mw_ac', 'capacity_mw_dc', 'mean_cf_ac', 'mean_cf_dc']
    hours_per_year = 8760
    # calculate solar supply per Tao's email. Gives MWh/year
    solar_df['generation_potential_ac'] = solar_df['capacity_mw_ac'] * solar_df['mean_cf_ac'] * hours_per_year
    solar_df['generation_potential_dc'] = solar_df['capacity_mw_dc'] * solar_df['mean_cf_dc'] * hours_per_year

    solar_sums_df = solar_df.groupby(location_cols, as_index=False).sum()

    # I think we just need capacity_mw_ac
    solar_keep_cols = location_cols + solar_supply_cols + ['generation_potential_ac', 'generation_potential_dc']
    solar_sums_df = solar_sums_df[solar_keep_cols]

    # rename fips column
    solar_sums_df = solar_sums_df.rename(columns={'cnty_fips':'FIPS'})

    # merge with land area. Need this because we want total land area, not just available solar land area in the NREL dataset
    km2_per_mi2 = 2.58999
    land_area_df = pd.read_csv(os.path.join(data_dir, '2024_Gaz_counties_national.txt'), delimiter='\t', dtype={'GEOID':str})
    land_area_df = land_area_df.rename(columns={'GEOID':'FIPS'})
    land_area_df['ALAND_SQKM'] = km2_per_mi2 * land_area_df['ALAND_SQMI']

    solar_sums_df = solar_sums_df.merge(land_area_df, on='FIPS')
    solar_sums_df['Solar_supply_per_land_area'] = solar_sums_df['generation_potential_ac'] / solar_sums_df['ALAND_SQKM']

    # keep only relevant columns
    solar_sums_df = solar_sums_df[['FIPS', 'state', 'county', 'capacity_mw_ac', 'capacity_mw_dc', 'Solar_supply_per_land_area', 'generation_potential_ac', 'generation_potential_dc', 'ALAND_SQKM']]

    # correlation between solar supply and land area? I bet there is
    solar_plot_df = solar_sums_df.copy()
    solar_plot_df['log solar supply'] = np.log(solar_plot_df['capacity_mw_ac'])
    solar_plot_df['log land area'] = np.log(solar_plot_df['ALAND_SQKM'])

    sns.regplot(solar_plot_df, x='log land area', y='log solar supply')
    plt.suptitle('Solar Supply vs Land Area')
    plt.savefig(os.path.join(graph_dir, 'solar_supply_land_area.png'))

    # ### NRI Weather hazards data
    # 
    # FEMA has created weather hazard index scores for various weather hazards. Right now, I'm only focusing on how these hazards affect agricultural losses. For AgPV, tornadoes are negative, since they destroy solar panels. Drought, hail, and heatwave are positives, since AgPV can help protect crops or diversify farmer income hagainst these hazards.
    # 
    # Relevant columns:
    # 
    # * TRND_ALRA: expected tornado ag loss rate
    # * HWAV_ALRA: heatwave expected ag loss rate
    # * HAIL_ALRA: hail expected ag loss rate
    # * DRGT_ALRA: drought expected loss
    # * RESL_SCORE: community resilience score
    # * SOVI_SCORE: sovial vulnerability score
    # * _EALA: Expected anual Loss
    # 
    # can use different suffixes for different metrics. Other metrics of interest:
    # * RISKV: risk value
    # * RISKS: risk index score
    # * RISKR: risk index rating
    # 
    # https://hazards.fema.gov/nri

    print('Getting Environmental Hazards...')
    nri_dir = 'NRI_Table_Counties'
    nri_file = 'NRI_Table_Counties.csv'

    nri_df = pd.read_csv(os.path.join(data_dir, nri_dir, nri_file), dtype={'STCOFIPS':str})

    # keep relevant columns
    prefixes = ['TRND', 'SWND', 'HWAV', 'HAIL', 'DRGT']
    suffixes = ['ALRA', 'EALA', 'RISKV', 'RISKS']
    nri_keep_cols = ['_'.join([pf, sf]) for pf in prefixes for sf in suffixes]

    nri_df = nri_df[['STCOFIPS', 'RESL_SCORE', 'SOVI_SCORE'] + nri_keep_cols]

    # rename relevant columns
    nri_df = nri_df.rename(columns={'STCOFIPS':'FIPS'})

    # ### Energy Burden
    # 
    # Percent of household income spent on energy. AgPV can help high burden counties lower their energy burden.
    # 
    # https://www.energy.gov/scep/slsc/lead-tool
    print('Getting Energy Burden...')
    eburden_file = 'LEAD Tool Data Counties.csv'

    eburden_df = pd.read_csv(os.path.join(data_dir, eburden_file), skiprows=range(0,8), dtype={'Geography ID':str})
    eburden_df = eburden_df.rename(columns={'Geography ID':'FIPS'})
    eburden_df = eburden_df[['FIPS', 'Energy Burden (% income)']]
    eburden_df.head()


    # ### R2R Data
    # 
    # Farmer income and % of minority owned cropland analyzed for Roads to Removal. Don't get too sad at the minority owned farm numbers.
    print('Getting farm demographics...')
    avg_farm_income_search = {'year':'2022',
                            'commodity_desc': 'INCOME, NET CASH FARM',
                            'domain_desc': 'TOTAL',
                            'agg_level': 'county',
                            'format': 'JSON',
                            'source_desc': 'census',
                            'statisticcat_desc': 'NET INCOME',
                            'short_desc': 'INCOME, NET CASH FARM, OF OPERATIONS - NET INCOME, MEASURED IN $ / OPERATION'
                            }

    clean_income_df = get_usda_economic_data(avg_farm_income_search, 'farm_income', fill_method=max)
    clean_income_df = clean_income_df[['FIPS', 'Value', 'Undisclosed_farm_income']]
    clean_income_df['Value'] = clean_income_df['Value'].astype(int)
    clean_income_df = clean_income_df.rename(columns={'Value':'Avg Farm Net Income ($)'})
    clean_income_df = clean_income_df.drop_duplicates(subset='FIPS', keep=False)

    # minority cropland
    minority_cropland_search = {'year':'2022',
                            'commodity_desc': 'AG LAND',
                            'domain_desc': 'PRODUCERS',
                            'agg_level': 'county',
                            'format': 'JSON',
                            'source_desc': 'census',
                            'short_desc': 'AG LAND, CROPLAND - AREA, MEASURED IN PCT OF AG LAND'}

    clean_minority_crop_df = get_usda_economic_data(minority_cropland_search, 'minority_crop', 'PRODUCERS', fill_method=0)
    clean_minority_crop_df = clean_minority_crop_df[['FIPS', 'domaincat_desc', 'Value', 'Undisclosed_minority_crop']]
    clean_minority_crop_df['Value'] = clean_minority_crop_df['Value'].astype(float)
    clean_minority_crop_df = clean_minority_crop_df.rename(columns={'Value':'Percent'})

    # Filter data and make calculations
    domaincats = ['PRODUCERS: (GENDER = FEMALE)', 'PRODUCERS: (RACE = WHITE, ALONE OR COMBINED WITH OTHER RACES)']
    filtered_minority_crop_df = clean_minority_crop_df.loc[clean_minority_crop_df['domaincat_desc'].isin(domaincats), :]
    filtered_minority_crop_df = filtered_minority_crop_df.pivot_table(values=['Percent', 'Undisclosed_minority_crop'], index='FIPS', columns='domaincat_desc')
    filtered_minority_crop_df[('Percent', 'Non-white')] =  100 - filtered_minority_crop_df[('Percent', 'PRODUCERS: (RACE = WHITE, ALONE OR COMBINED WITH OTHER RACES)')]
    minority_crop_percents = filtered_minority_crop_df['Percent']
    minority_crop_percents = minority_crop_percents.rename(columns={'PRODUCERS: (GENDER = FEMALE)': 'Percent Woman-owned Cropland',
                                                                    'Non-white': 'Percent Non-white Cropland'})
    minority_crop_percents = minority_crop_percents.drop(columns=['PRODUCERS: (RACE = WHITE, ALONE OR COMBINED WITH OTHER RACES)'])
    minority_crop_percents = minority_crop_percents.fillna(0)
    minority_crop_percents = minority_crop_percents.reset_index()

    # pastureland
    pastureland_search = {'year':'2022',
                        'commodity_desc': 'AG LAND',
                        'domain_desc': 'TOTAL',
                        'sector_desc': 'ECONOMICS',
                        'group_desc_LIKE': 'FARM & LAND & ASSETS',
                        'agg_level': 'county',
                        'format': 'JSON',
                        'source_desc': 'census',
                        #'statisticcat_desc':'AREA',
                        'short_desc__LIKE': 'AG LAND, PASTURE'}

    clean_pastureland_df = pd.read_csv(os.path.join(data_dir, 'pastureland.csv'), dtype={'State ANSI':str, 'County ANSI': str})
    clean_pastureland_df = clean_usda_values(clean_pastureland_df, 'pastureland')
    clean_pastureland_df['FIPS'] = clean_pastureland_df['State ANSI'] + clean_pastureland_df['County ANSI']
    clean_pastureland_df = clean_pastureland_df.loc[clean_pastureland_df['Data Item'] == 'AG LAND, PASTURELAND, (EXCL CROPLAND & WOODLAND) - ACRES']
    clean_pastureland_df = clean_pastureland_df[['FIPS', 'Value', 'Undisclosed_pastureland']]
    clean_pastureland_df['Value'] = clean_pastureland_df['Value'].astype(float)
    clean_pastureland_df = clean_pastureland_df.rename(columns={'Value':'Pastureland'})
    clean_pastureland_df = clean_pastureland_df.drop_duplicates(subset='FIPS', keep=False)

    # # Energy Information Admisitration: Coop data from 2022
    # Counties with strong existing cooperatively owned energy generators could have a stronger foundationg for equitably adopting AgPV
    # 
    # Unfortunately, the data from EIA doesn't have county FIPS codes. They only have state abbreviations and county names. We'll have to do some joins to get the actual FIPS code.
    print('Getting EIA ownership data...')
    eia_dir = 'EIA_f8612022'
    meters_file = 'Advanced_Meters_2022.xlsx'
    sales_file = 'Sales_Ult_Cust_2022.xlsx'
    service_territory_file = 'Service_Territory_2022.xlsx'

    eia_sales_df = pd.read_excel(os.path.join(data_dir, eia_dir, sales_file), header=[0,1,2], nrows=2829)
    eia_locations_df = pd.read_excel(os.path.join(data_dir, eia_dir, service_territory_file))

    # keep only totals column from sales
    eia_sales_df = eia_sales_df.loc[:, eia_sales_df.columns.get_level_values(0).isin(['Utility Characteristics', 'TOTAL'])]
    eia_sales_df.columns = eia_sales_df.columns.get_level_values(-1)
    eia_sales_df = eia_locations_df[['Utility Number', 'County']].merge(eia_sales_df, on='Utility Number')

    # Clean case just in case there's some shenanigans
    eia_sales_df['State'] = eia_sales_df['State'].apply(lambda x: x.strip().upper())
    eia_sales_df['County'] = eia_sales_df['County'].apply(lambda x: x.strip().title())

    # get county geo info
    state_abbrevs_file = 'us-states-territories.csv'
    county_file = 'US_FIPS_Codes.xls'
    state_df = pd.read_csv(os.path.join(data_dir, state_abbrevs_file), encoding='unicode_escape')
    state_df['Name'] = state_df['Name'].apply(lambda x: x.strip().title())

    county_df = pd.read_excel(os.path.join(data_dir, county_file), header=1, dtype={'FIPS State':str, 'FIPS County':str})
    county_df['State'] = county_df['State'].apply(lambda x: x.strip().title())
    county_df['FIPS'] = county_df['FIPS State'] + county_df['FIPS County']

    state_county_df = state_df[['Name', 'Abbreviation', 'area (square miles)']].merge(county_df[['State', 'County Name', 'FIPS']], left_on='Name', right_on='State')
    state_county_df = state_county_df[['State', 'Abbreviation', 'County Name', 'FIPS', 'area (square miles)']]

    state_county_df['Abbreviation'] = state_county_df['Abbreviation'].apply(lambda x: x.strip().upper())
    state_county_df['County Name'] = state_county_df['County Name'].apply(lambda x: x.strip().title())
    state_county_df = state_county_df.rename(columns={'County Name':'County'})

    eia_sales_df = eia_sales_df.merge(state_county_df, left_on=['State', 'County'], right_on=['Abbreviation', 'County'])
    keep_cols = ['FIPS', 'Ownership', 'area (square miles)', 'Thousand Dollars', 'Megawatthours', 'Count']
    eia_sales_df = eia_sales_df[keep_cols]

    # get percentage of non-investor-owned energy
    sales_sum_df = eia_sales_df.loc[:, ['FIPS', 'Thousand Dollars', 'Megawatthours', 'Count']].groupby('FIPS', as_index=False).sum()

    eia_ownership_sales_df = eia_sales_df.loc[:, ['FIPS', 'Ownership', 'Thousand Dollars', 'Megawatthours', 'Count']].groupby(['FIPS', 'Ownership'], as_index=False).sum()
    sales_sum_df = eia_ownership_sales_df.merge(sales_sum_df, on='FIPS', suffixes=('', '_total'))
    sales_sum_df['Percent Megawatthours'] = 100 * sales_sum_df['Megawatthours'] / sales_sum_df['Megawatthours_total']
    sales_sum_df['Percent Count'] = 100 * sales_sum_df['Count'] / sales_sum_df['Count_total']

    eia_percent_df = sales_sum_df.loc[sales_sum_df['Ownership'] == 'Investor Owned', :]
    eia_percent_df['Percent Non Investor Owned'] = 100 - eia_percent_df['Percent Count']
    eia_percent_df = eia_percent_df[['FIPS', 'Percent Non Investor Owned']]

    # # Carbon Intensity of Electricity
    # Table 4. Per capita energy-related carbon dioxide emissions by state
    # 
    # metric tons of energy-related carbon dioxide per person
    # 
    # https://www.eia.gov/environment/emissions/state/
    print('Getting State Carbon Intensity...')
    c_intensity_df = pd.read_excel(os.path.join(data_dir, 'table4.xlsx'), header=4)

    c_intensity_df = c_intensity_df[['State', 2022]]
    c_intensity_df = c_intensity_df.rename(columns={2022:'tonne_co2_per_person'})

    # need to get state FIPS info from solar sums df
    county_c_intensity_df = solar_sums_df.merge(c_intensity_df, left_on = 'state', right_on='State')
    county_c_intensity_df = county_c_intensity_df[['FIPS', 'tonne_co2_per_person']]

    # ### Ag Worker Heat Risk
    # This is likely to be highly correlated with the existing ag heat risk index. the plan here is to:
    # 
    # * get excessive heat day change due to climate change
    #     * https://ephtracking.cdc.gov/DataExplorer/?query=c380f325-ff71-4070-ba52-839272757eae
    # * multiply by qwi employment in crop and animal production to get the number of worker days exposed to excessive heat due to climate change
    #     * Census QWI data
    print('Getting Ag Worker Heat Risk...')
    excessive_heat_df = pd.read_csv(os.path.join(data_dir, 'excessive_heat_days_change_GWL', 'data_143828.csv'), dtype={'CountyFIPS':str, 'StateFIPS':str})
    excessive_heat_df = excessive_heat_df.drop(columns='Unnamed: 7')
    excessive_heat_df = excessive_heat_df.rename(columns = {'CountyFIPS':'FIPS'})

    state_fips = excessive_heat_df['StateFIPS'].unique()
    qwi = qwi_utils(qwi_key)

    print('\tGetting Ag jobs...')
    ag_jobs_dfs = [get_qwi_dfs(qwi, state) for state in state_fips]
    ag_jobs_df = pd.concat(ag_jobs_dfs)

    # clean ag jobs data
    ag_jobs_df = ag_jobs_df.dropna(subset='Emp')
    ag_jobs_df['Emp'] = ag_jobs_df['Emp'].astype(int)
    ag_jobs_df = ag_jobs_df.groupby(['state','county', 'time'], as_index=False).sum()
    ag_jobs_df = ag_jobs_df[['state', 'county', 'Emp']].groupby(['state','county'], as_index=False).mean()
    ag_jobs_df['FIPS'] = ag_jobs_df['state'] + ag_jobs_df['county']
    ag_jobs_df = ag_jobs_df[['FIPS', 'Emp']]

    # calculate ag-worker days
    worker_heat_days_df = excessive_heat_df.merge(ag_jobs_df, on='FIPS')
    worker_heat_days_df['ag_worker_heat_days'] = worker_heat_days_df['Value'] * worker_heat_days_df['Emp']
    worker_heat_days_df = worker_heat_days_df[['FIPS', 'ag_worker_heat_days']]
    worker_heat_days_df.head()

    print('Merging All Data Together...')
    all_dfs = [solar_sums_df, 
            nri_df, 
            eburden_df, 
            eia_percent_df, 
            crops_sum_df, 
            sheep_goats_clean_df,
            sheep_goats_demo_df,
            clean_income_df,
            minority_crop_percents,
            clean_pastureland_df,
            county_c_intensity_df,
            solar_land_df,
            worker_heat_days_df
            ]

    all_dfs = [df.dropna(subset='FIPS').set_index('FIPS') for df in all_dfs]
    merged_df = pd.concat(all_dfs, axis=1)

    # format undisclosed columns
    for col in merged_df.columns:
        if re.match(r'^undisclosed_', col.lower()):
            merged_df[col] = merged_df[col] > 0
            merged_df = merged_df.rename(columns={col:'Has_' + col.lower()})
            
    # label counties that nave neither livestock nor crop compatibility
    merged_df['Ag_compatibility_exists'] = ~merged_df[['AgPV_crop_totals', 'Sheep_goats_total_sales']].isna().all(axis=1)

    zero_fill_cols = ['Percent Non-white Cropland',
                    'Percent Woman-owned Cropland',
                    'Percent Non Investor Owned',
                    'AgPV_crop_totals', 
                    'Sheep_goats_total_sales', 
                    'Percent Nonwhite Sheep Goats Producers',
                    'Percent Woman-owned Sheep Goats Producers',
                    'Avg Farm Net Income ($)',
                    'Pastureland',
                    'ag_worker_heat_days']

    merged_df[zero_fill_cols] = merged_df[zero_fill_cols].fillna(0)

    env_risks = {'negative': ['TRND_EALA'],
                'positive': ['HWAV_EALA', 'HAIL_EALA', 'DRGT_EALA']
                }
    eala_sums = merged_df[env_risks['positive']].sum(axis=1)
    net_eala = eala_sums - merged_df[env_risks['negative']].sum(axis=1)

    merged_df['eala_sum_positive'] = eala_sums
    merged_df['eala_net'] = net_eala

    merged_df.to_csv(os.path.join(data_dir, 'agpv_index_variables.csv'))

    print('All Data Gathered!')
    print(merged_df.head())

    # does Hawaii exist
    hi_counties = ['15001', '15003', '15005', '15007', '15009']
    print('Hawaii Index Vars:')
    print(merged_df.loc[merged_df.index.isin(hi_counties), :])
    # solar_df.loc[solar_df.cnty_fips.isin(hi_counties), :]

    # does alaska exist in data?
    print('AK Index Vars:')
    print(merged_df.loc[merged_df.state == 'Alaska', :])
    print('AK solar:')
    solar_sums_df.loc[solar_sums_df['state'] == 'Alaska', :]

    ak_missing_count = 0
    ak_exists_count = 0

    for df in all_dfs:
        print(df.columns)
        states = {fips[:2] for fips in df.index}
        if '02' in states:
            print('Ak Exists')
            ak_exists_count += 1
        else:
            print('No Ak Found')
            ak_missing_count += 1
            
    print(f'AK exists cols: {ak_exists_count}')
    print(f'AK missing cols: {ak_missing_count}')