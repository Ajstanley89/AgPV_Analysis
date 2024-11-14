import pandas as pd
import geopandas as gpd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import oapackage
import matplotlib.colors as mcolor

from adjustText import adjust_text

sns.set_style('darkgrid')

data_dir = 'AgPV_data'
export_dir = 'Export_data'


# ### Tribal Counties
# Get list of counties that intersect with tribal lands
tribes_gdf = gpd.read_file(os.path.join(data_dir, 'tl_2024_us_aiannh.zip'))
counties_gdf = gpd.read_file(os.path.join(data_dir, 'tl_2024_us_county.zip'))

tribe_counties_gdf = tribes_gdf.overlay(counties_gdf)

tribal_county_fips = (tribe_counties_gdf['STATEFP'] + tribe_counties_gdf['COUNTYFP']).unique()
# ### Index Visualizations
df = pd.read_csv(os.path.join(export_dir, 'agpv_index_v7.csv'), dtype={'FIPS':str, 'state':str, 'county':str})
df['County, State'] = df['county'] + ', ' + df['state']
df.columns

class agpvVisualizer():
    def __init__(self, tribal_fips=tribal_county_fips, df: pd.DataFrame=df, save_dir='AgPV_graphs'):
        self.tribal_fips = tribal_fips
        self.df = df
        self.save_dir = save_dir
        
    def make_bar_chart(self, x_label, title_start = 'Solar Supply', top_n=20, x='generation_potential_ac', 
                       initial_sort_value='Theme_Weighting_Score', logx=False, graph_kwargs={}):
        """Makes the solar supply of top cobenefits index bar charts"""
        top_df = self.df.sort_values(initial_sort_value, ascending=False).iloc[:top_n, :]
        
        plot_order_df = top_df.sort_values(x, ascending=False)

        fig, ax = plt.subplots(1,1, figsize=(10,10), dpi=300)
        g = sns.barplot(plot_order_df, x=x, y='County, State', ax=ax, **graph_kwargs)
        plt.suptitle(f'{title_start.title()} of Top {top_n} Co-benefits Scores')
        plt.tight_layout()

        plt.vlines(top_df[x].mean(), top_df.sort_values(x, ascending=False)['County, State'].iloc[-1], top_df.sort_values(x, ascending=False)['County, State'].iloc[0],
                   color='k', linestyle='--', label='National Mean')

        # set fill pattern for tribal land
        bars = ax.patches

        for fips, bar in zip(plot_order_df.FIPS, bars):
            if fips in self.tribal_fips:
                bar.set_hatch('//')
                bar.set_label('Tribal Land')

        ax.set_xlabel(x_label.title())

        # remove duplicate labels
        h, l = ax.get_legend_handles_labels()
        by_label = dict(zip(l, h))
        ax.legend(by_label.values(), by_label.keys())   

        if logx:
            plt.xscale('log')     
        
        plt.savefig(os.path.join(self.save_dir, f'{x}_bar_chart_top_{top_n}.png'))
        
        plt.show()
        
        return g
        
    def make_scatter_plot(self, suptitle, x='generation_potential_ac', hue='Solar_supply_per_land_area', 
                          y='Theme_Weighting_Score', size='AgPV_crop_totals', annot_col='Theme_Weighting_Score',
                          palette=sns.color_palette("YlOrBr", as_cmap=True),
                          top_n = 5, logx=False, logy=False, savefig=True, graph_kwargs={}, annot_kwargs={}):
        
        scatter_df = self.df.copy()        

        fig, ax = plt.subplots(1,1, figsize=(10,10), dpi=300)
        g = sns.scatterplot(scatter_df, x=x, y=y, size=size, sizes=(10,300), hue=hue, palette=palette, edgecolor='k', alpha=0.75, ax=ax, **graph_kwargs)

        top_df = scatter_df.sort_values(annot_col, ascending=False).iloc[:top_n, :]

        def annotate_counties(row, x=x, y=y, g=g, arrowprops={'arrowstyle':'-', 'color': 'black'}, annot_kwargs=annot_kwargs) -> None:
            text = f"{row['county']}, {row['state']}"
            xy = xy=(row[x], row[y])
            
            g.annotate(text=text, xy=xy, arrowprops=arrowprops, size=5,
                      bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'), **annot_kwargs)

        top_df.apply(lambda row: annotate_counties(row, annot_kwargs=annot_kwargs), axis=1)

        if logy:
            plt.yscale('log')
        if logx:
            plt.xscale('log')

        plt.suptitle(suptitle)
        plt.tight_layout()
        
        if savefig:
            plt.savefig(os.path.join(self.save_dir, f'{suptitle}_scatter_top_{top_n}.png'))
                
        return g
        
    def set_data(self, new_df):
        self.df = new_df
        
        return self
        
    def set_save_dir(self, new_save_dir):
        self.save_dir = new_save_dir
        
        return self
    
    def calculate_col_percentile(self, col: str):
        """Determines the percentile rank of a df col and adds that as new col to the df"""
        self.df[col + '_percentile'] = self.df[col].rank(pct=True)
        
        return self

if __name__ == '__main__':
    x_label = 'Power Generation Potential (MWhr/year)'

    agpviz = agpvVisualizer()

    agpviz.make_bar_chart(x_label)

    x = 'Solar_supply_per_land_area'
    x_label = 'Power Generation Potential Per Area (MWhr/km2/year)'

    agpviz.make_bar_chart(x_label, x=x, title_start='Solar Supply Per Area')

    # ### Scatterplot
    # 
    # Kim's suggestion
    # 4.	Looking at both scatterplots, I think this might be the ideal between all of them: Equity index on Y-Axis (what the ARER journal review said is that readers just look for stuff that’s at the ‘top’ of the graph for clarity), Solar Capacity on X-Axis, AgPV crop totals = size of bubbles, and a yellow color ramp where the darkest yellow indicates the counties that are producing the most solar energy per land area, but the counties highlighted with names are those with the highest equity index for AgPV. So a reader should be able to look at the graph and see which of those ‘named counties’ (which are also in the bar chart) have the darkest yellow with greatest capacity. Again… probably should find some way to identify tribal counties, dunno how though… Let’s see how this looks? 

    agpviz.make_scatter_plot('AgPV: Equity Index, Solar Potential, Crops', logx=True, annot_kwargs={'xytext':(10, 1), 'textcoords':'offset points'})


    agpviz.calculate_col_percentile('Theme_Weighting_Score')

    annot_kwargs={'xytext':(10, 10), 'textcoords':'offset points'}

    scatter_params = {'palette': sns.diverging_palette(145, 300, s=60, as_cmap=True),
                    'y': 'AgPV_crop_totals',
                    'size':'eala_net',
                    'annot_col': 'eala_net',
                    'hue': 'Theme_Weighting_Score_percentile',
                    'logx': True,
                    'logy': True,
                    'annot_kwargs': annot_kwargs,
                    'top_n': 5}

    agpviz.make_scatter_plot('Ag PV Crop Potential and Ag Loss', **scatter_params)

    agpviz.calculate_col_percentile('Theme_Weighting_Score')

    annot_kwargs={'xytext':(10, 10), 'textcoords':'offset points'}

    scatter_params = {'palette': sns.diverging_palette(145, 300, s=60, as_cmap=True),
                    'y': 'Sheep_goats_total_sales',
                    'size':'eala_net',
                    'annot_col': 'eala_net',
                    'hue': 'Theme_Weighting_Score_percentile',
                    'logx': True,
                    'logy': True,
                    'annot_kwargs': annot_kwargs,
                    'top_n': 5}

    agpviz.make_scatter_plot('Ag PV Livestock, Solar Potential, and Environmental Losses', **scatter_params)

