from requests import get
import pandas as pd

class qwi_utils():
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def build_qwi_url(self, **params) -> str:
        """
        Takes a dict of search params and formats them into the qwick stats url search query
        """
        assert('state' in params.keys())
        base_url = 'https://api.census.gov/data/timeseries/qwi/sa?'
        county_url = f'get=Emp&for=county:*&in=state:{params.get("state")}'

        query_url = base_url + county_url

        for key, value in params.items():
            if key.lower() != 'state':
                indicator = f'&{key}='
                if isinstance(value, str):
                    query_url += indicator + value

                elif isinstance(value, list):
                    query_url += indicator + indicator.join(value)
                else:
                    print(f'Could not format {key}: {value}')

        return query_url + '&key=' + self.api_key


    def get_qwi_data(self, **params):
        '''returns df of qwik search json results'''
        r =  get(self.build_qwi_url(**params))
        try:
            return r
        
        except Exception as e:
            print(f"Request is getting this error: {e}")
            print(r.text)

    def make_qwi_df(self, **params) -> pd.DataFrame:
        r = self.get_qwi_data(**params)
        try:
            j = r.json()
            # column names are in first row
            columns = j[0]
            # actual data is in the other rows
            data = j[1:]

            return pd.DataFrame(data=data, columns=columns)
        
        except Exception as e:
            print(f"State: {params.get('state')}")
            print(e)
            print(r)
            print(r.url)
            print(r.text)
            return None

if __name__ == '__main__':
    from config import qwi_key

    qwi_utils = qwi_utils(qwi_key)

    test_state = '06' # California
    test_naics = ['111', '112'] # crop production and animal production

    search_params1 = {'state': test_state,
                      'time': '2022',
                      'industry': '111'
                      }
    
    search_params2 = {'state': test_state,
                      'time': '2022',
                      'industry': test_naics
                      }

    df1 = qwi_utils.make_qwi_df(**search_params1)
    df2 = qwi_utils.make_qwi_df(**search_params2)

    print(df1.head())
    print(df2['industry'].value_counts())