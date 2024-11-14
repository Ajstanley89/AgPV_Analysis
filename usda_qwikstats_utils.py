from requests import get
import pandas as pd

class usda_qwikstats_utils():
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def build_qwikstats_url(self, search_param_dict: dict) -> str:
        """
        Takes a dict of search params and formats them into the qwick stats url search query
        """
        base_qwik_stats_url = f'https://quickstats.nass.usda.gov/api/api_GET/?key={self.api_key}&'
        search_param_str = '&'.join([key + '=' + value for key, value in search_param_dict.items()])
        return base_qwik_stats_url + search_param_str

    def get_qwikstats_data(self, search_params):
        '''returns df of qwik search json results'''
        r =  get(self.build_qwikstats_url(search_params))
        try:
            return r
        
        except Exception as e:
            print(f"Request is getting this error: {e}")
            print(r.text)