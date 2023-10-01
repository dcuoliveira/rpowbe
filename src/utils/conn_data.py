import os
import json
import pickle
import pandas as pd
from tqdm import tqdm

def save_csv_result_in_blocks(df, args, path):

    if not os.path.exists(path):
        os.makedirs(path)

    df["date"] = pd.to_datetime(df["date"])
    years = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

    for y in tqdm(years, total=len(years), desc="Saving Results"):
        
        tmp_results = {

            "train_loss": None,
            "eval_loss": None,
            "test_loss": None,
            "returns": None,
            "weights": None,
            "summary": df.loc[pd.to_datetime(df["date"]).year == y]

        } 

        # save results
        save_pickle(obj=tmp_results, path=os.path.join(path, "results_{}.pickle".format(y)))
        tmp_results["summary"].to_csv(os.path.join(path, "summary_{}.csv".format(y)), index=False)

def save_result_in_blocks(results, args, path):

    if not os.path.exists(path):
        os.makedirs(path)

    # save args
    args_dict = vars(args)  
    with open(os.path.join(path, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp)

    years = list(pd.Series([dtref.year for dtref in results["summary"]["date"]]).unique())
    
    if (results["means"] is not None) or (results["covs"] is not None):
        tot = results["means"].shape[0] if results["means"] is not None else results["covs"].shape[0]
        start = 0
        end =  parts = tot // len(years)

    for y in tqdm(years, total=len(years), desc="Saving Results"):
        
        tmp_results = {

            "train_loss": None,
            "eval_loss": None,
            "test_loss": None,
            "returns": results["returns"].loc[results["returns"].index.year == y],
            "weights": results["weights"].loc[results["weights"].index.year == y],
            "summary": results["summary"].loc[results["summary"]["date"].dt.year == y]

        } 

        # save results
        save_pickle(obj=tmp_results, path=os.path.join(path, "results_{}.pickle".format(y)))
        tmp_results["summary"].to_csv(os.path.join(path, "summary_{}.csv".format(y)), index=False)

        if (results["means"] is not None) and (results["covs"] is not None):

            if (results["means"] is not None):
                tmp_means = {

                    "means": results["means"][start:end],
                }
                # save_pickle(obj=tmp_means, path=os.path.join(path, "means_{}.pickle".format(y)))

            if (results["covs"] is not None):
                tmp_covs = {

                    "covs": results["covs"][start:end]
                }
                # save_pickle(obj=tmp_covs, path=os.path.join(path, "covs_{}.pickle".format(y)))

            start = end
            end += parts

            if end > tot:
                end = tot

    if results["test_loss"] is not None:
        save_pickle(obj={"test_loss": results["test_loss"]}, path=os.path.join(path, "test_loss.pickle")) 

def save_pickle(path: str,
                obj: dict):

    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: str):

    with open(path, 'rb') as handle:
        target_dict = pickle.load(handle)

    return target_dict