import yaml
import requests
import pandas as pd
import io
from tqdm import notebook
from tqdm import tqdm
import time
import os


def get_sp500(url: str) -> pd.DataFrame:
    table=pd.read_html(url)
    df = table[0]
    df['Symbol'] = df["Symbol"].apply(lambda x: x.replace('.', '-'))
    df.to_csv('dataset/info_data/S&P500-Info.csv')
    df.to_csv("dataset/info_data/S&P500-Symbols.csv", columns=['Symbol'])
    return df


def get_timeseries(*config) -> None:
        function, config, output_path, symbols = config
        for i, ticker in enumerate(notebook.tqdm(symbols.loc[:, "Symbol"])):
            request_params = {
                    "symbol": ticker,
                    "apikey": config["KEY"],
                    "function": function
                }
            if config["FUNCTIONS"][function]["FETCHING"]:
                request_params.update(config["FUNCTIONS"][function]["FETCHING"])

            while True:
                response = requests.get(
                    config["API_URL"],
                    request_params
                )
                if response.ok:
                    new_df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
                    if len(new_df) == 2:
                        print(f"For symbol {ticker}")
                        print(f"Response deined because {pd.read_csv(io.StringIO(response.content.decode('utf-8')))}")
                        print("Sleeping 10s to recover!")
                        time.sleep(10)
                    else:
                        break
                else:
                    print(f"Response returned with code {response.status_code}")
                    print("Sleeping 2s to recover!")
                    time.sleep(2)
        
            new_df.insert(0, "symbol", ticker)
            if i != 0:
                new_df.to_csv(output_path, mode='a', header=False, index=False)
            else:
                new_df.to_csv(output_path, header=True, index=False)



def get_fundamentals(*config) -> None:
        function, config, output_path, symbols = config
        for i, ticker in enumerate(notebook.tqdm(symbols.loc[:, "Symbol"])):
            request_params = {
                    "symbol": ticker,
                    "apikey": config["KEY"],
                    "function": function
                }
            if config["FUNCTIONS"][function]["FETCHING"]:
                request_params.update(config["FUNCTIONS"][function]["FETCHING"])

            while True:
                response = requests.get(
                    config["API_URL"],
                    request_params
                )
                if response.ok:
                    data = response.json()
                    if "Information" in data:
                        print(f"For symbol {ticker}")
                        print(f"Response deined because {response.json()['Information']}")
                        print("Sleeping 10s to recover!")
                        time.sleep(10)
                    else:
                        if config["FUNCTIONS"][function]["PROCESSING"]:
                            new_df = pd.DataFrame(data[config["FUNCTIONS"][function]["PROCESSING"]["extract"]]) 
                        else:
                            new_df = pd.DataFrame([data])
                        break

                else:
                    print(f"Response returned with code {response.status_code}")
                    print("Sleeping 2s to recover!")
                    time.sleep(2)

            new_df.insert(0, "symbol", ticker)
            if i != 0:
                new_df.to_csv(output_path, mode='a', header=False, index=False)
            else:
                new_df.to_csv(output_path, header=True, index=False)

def get_economy(*config) -> None:
        function, config, output_path, _ = config
        request_params = {
                "apikey": config["KEY"],
                "function": function
            }
        if config["FUNCTIONS"][function]["FETCHING"]:
            request_params.update(config["FUNCTIONS"][function]["FETCHING"])

        while True:
            response = requests.get(
                config["API_URL"],
                request_params
            )
            if response.ok:
                new_df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
                if new_df.iloc[0, 0] == "Information":
                    print(f"Response deined because {pd.read_csv(io.StringIO(response.content.decode('utf-8'))).iloc[0, 1]}")
                    print("Sleeping 10s to recover!")
                    time.sleep(10)
                else:
                    break
            else:
                print(f"Response returned with code {response.status_code}")
                print("Sleeping 2s to recover!")
                time.sleep(2)

        new_df.to_csv(output_path, header=True, index=False)


def get_sentiment(*config) -> None:
    function, config, output_path, symbols = config
    time_file = "dataset/raw_data/timeseries/time_series_monthly_adjusted.csv"
    time_df = pd.read_csv(time_file)
    
    for i, ticker in enumerate(notebook.tqdm(symbols.loc[:, "Symbol"])):
        current_time_df = (
            time_df[time_df["symbol"] == ticker]
            .loc[:, "timestamp"]
            .head(60)
        )
        for j in notebook.tqdm(range(len(current_time_df)-1), leave=False):
            request_params = {
                    "tickers": ticker,
                    "apikey": config["KEY"],
                    "function": function,
                    "time_from": "".join((current_time_df.iloc[j+1]).split('-')) + "T2359",
                    "time_to": "".join((current_time_df.iloc[j]).split('-')) + "T2359"
                }
            if config["FUNCTIONS"][function]["FETCHING"]:
                request_params.update(config["FUNCTIONS"][function]["FETCHING"])

            while True:
                response = requests.get(
                    config["API_URL"],
                    request_params
                )
                if response.ok:
                    data = response.json()
                    if "Information" in data:
                        if "No articles found" in data['Information']:
                            relevant_data = [{
                                "ticker": ticker,
                                "relevance_score": None,
                                "ticker_sentiment_score": None,
                                "ticker_sentiment_label": None,
                                "timestamp": current_time_df.iloc[j+1]
                            }]
                            new_df = pd.DataFrame(relevant_data)
                            break
                        else:
                            print(f"For symbol {ticker}")
                            print(f"Response deined because {data['Information']}")
                            print("Sleeping 10s to recover!")
                            time.sleep(10)
                    else:
                        relevant_data = []
                        for mention in (
                            data[config["FUNCTIONS"][function]["PROCESSING"]["extract"]]
                        ):
                            for details in mention[config["FUNCTIONS"][function]["PROCESSING"]["details"]]:
                                if details["ticker"] == ticker:
                                    details.update({
                                        "timestamp": current_time_df.iloc[j]
                                    })
                                    relevant_data.append(details)
                        new_df = pd.DataFrame(relevant_data)
                        break

                else:
                    print(f"Response returned with code {response.status_code}")
                    print("Sleeping 2s to recover!")
                    time.sleep(2)

            if i == 0 and j==0:
                new_df.to_csv(output_path, header=True, index=False)
            else:
                new_df.to_csv(output_path, mode='a', header=False, index=False)
                

FUNCTIONS = {
    "timeseries": get_timeseries,
    "fundamentals": get_fundamentals,
    "sentiment": get_sentiment,
    "economy": get_economy
}

def extract_data(config_file: str) -> None:
    with open(config_file) as config_fh:
        try:
            config = yaml.safe_load(config_fh)["API_CONNECTION"]
        except yaml.YAMLError as e:
            print(f"An error occured when reading YAML file: {e}")

    symbols = get_sp500(config["SYMBOL_URL"])

    path = "dataset/raw_data"

    for key in FUNCTIONS.keys():
        if not os.path.isdir(f"{path}/{key}"):
            os.mkdir(f"{path}/{key}")

    for function in config["FUNCTIONS"].keys():
        output_path = f"{path}/{config['FUNCTIONS'][function]['TAG']}/{function.lower()}.csv"
        if config["UPDATE_ALL"]:
            print(f"Fetching data for {function}")
            FUNCTIONS[config["FUNCTIONS"][function]["TAG"]](function, config, output_path, symbols)
        elif os.path.isfile(output_path):
            print(f"Data already retrived for {function}")
        else:
            print(f"Fetching data for {function}")
            FUNCTIONS[config["FUNCTIONS"][function]["TAG"]](function, config, output_path, symbols)