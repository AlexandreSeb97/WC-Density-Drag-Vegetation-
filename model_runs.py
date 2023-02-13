# Script to run library of models for all sites and configurations, outputting them to the Model Runs folder
# Built by Alexandre Erich Sebastien Georges, under Mark T. Stacey
# University of California, Berkeley - Civil and Environmental Engineering, 2022


# Import water column model class and components
from water_column_model.column import Column
from water_column_model.advance import *
from water_column_model.params import *
from water_column_model.params import A,B,C,E
# Data Science and Visualization Imports
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import xarray as xr
import numpy as np
# Miscellaneous
from tqdm import tqdm

# Importing USGS Data
filename = 'data/USGS_veg_density_09.16.2022.csv'
df = pd.read_csv(filename)

# Setting max water column height (arbitrary for now, but based on maximum vegetation height from data)
H = 2.8072 # centimeters
# Adjusting Tide level to topography (elevation) for correct water column height for each site
df['wc_height'] = H - df['Elevation']
# New dataframe without unneccessary columns
wcolumns = df.drop(['plot_ID', 'frontal_5', 'frontal_10', 'frontal_15', 'frontal_20', 'frontal_20', 'frontal_25', 'frontal_30'], axis=1)
# Pulling parameters
params = [A, B, C, Sq, kappa, nu, g, rho0, alpha]

# Count of model runs for the collection, will be used to label individual runs

# for a given set of parameteres, create a column, run the model and return Flow Properties
def run_column_model(N, H, L, SMALL, params, veg_density, veg_ht, density_alpha):
    # Initializing Column Object
    col = Column(N, H, L, SMALL)
    A = params[0]
    B = params[1]
    C = params[2]
    Sq = params[3]
    kappa = params[4]
    nu = params[5]
    g = params[6]
    rho0 = params[7]
    alpha = params[8]
    # Setting up vlaues and importing Vegetation Distribution from Data
    col = col.setup(A, B, C, Sq, kappa, SMALL, nu, g, rho0, alpha)
    col = col.import_veg(density_alpha, veg_density, veg_ht)
    # Running Model for Col.M amount of time (See Column Class to change M)
    t = [] 
    for i in tqdm(range(col.M), leave=False):
        t.append(col.dt*(i))
        # Unew, Cnew, Qnew, Q2new, Q2Lnew, rhonew, Lnew, nu_tnew, Kznew, Kqnew, N_BVnew, N_BVsqnew
        [col.U, col.scalar, col.Q, col.Q2, col.Q2L, col.rho, col.L, col.nu_t, col.Kz, col.Kq, col.N_BV, col.N_BVsq] = wc_advance(col, t_px, px0, t[i])
        # Have TQDM instead print('Step! t=' +str(t[i])+'s')
    ### Only for now ### Return Velocity Profile ### Will implement returning more later
    col_res = pd.DataFrame({'U Velocity':col.U,
                            'Q':col.Q,
                            'Q2':col.Q2,
                            'Q2L':col.Q2L,
                            'Z':col.z,
                            'Kq':col.Kq,
                            'nu_t':col.nu_t,
                            })
    return col_res

def run_collection(n_runs):
    
    export_baseline = 'model_runs/baseline_wc_full.nc'
    export_runs = 'model_runs/'

    # Different alpha will be asked for every new run
    alpha_list = list(map(float,input("\nInput " + str(n_runs) + " LIST of values [0-1] for alpha density coefficient: : ").strip().split()))[:n_runs]
    
    # Making new dataframes for specific sites
    wcol_bay = wcolumns[wcolumns['site'] == 'Bay']
    wcol_int = wcolumns[wcolumns['site'] == 'Interior']
    wcol_creek = wcolumns[wcolumns['site'] == 'Creek']

    # Running and exporting water column model for base case
    col_res_base = wcolumns.apply(lambda x: run_column_model(80,(x.wc_height*100),0,SMALL,params,x.density_final,x.ave_ht,density_alpha=0), axis=1)
    ds_base = xr.concat([df.to_xarray() for df in col_res_base], dim="Model Runs")
    ds_base = ds_base.assign_attrs(description='Base Model run results for all sites.')    
    ds_base.to_netcdf(path=export_baseline, mode='w')

    # Running and exporting water column model for vegetated cases
    for i in range(n_runs):
        
        alpha_ask = alpha_list[i]
        # Initializing and running water column model for each entry
        col_res_all = wcol_bay.apply(lambda x: run_column_model(80,(x.wc_height*100),0,SMALL,params,x.density_final,x.ave_ht,alpha_ask), axis=1)
        col_res_bay = wcol_bay.apply(lambda x: run_column_model(80,(x.wc_height*100),0,SMALL,params,x.density_final,x.ave_ht,alpha_ask), axis=1)
        col_res_creek = wcol_creek.apply(lambda x: run_column_model(80,(x.wc_height*100),0,SMALL,params,x.density_final,x.ave_ht,alpha_ask), axis=1)
        col_res_interior = wcol_int.apply(lambda x: run_column_model(80,(x.wc_height*100),0,SMALL,params,x.density_final,x.ave_ht,alpha_ask), axis=1)

        # Turning results into xArray for easy visualization and manip
        ds_all = xr.concat([df.to_xarray() for df in col_res_all], dim="Model Runs")
        ds_all = ds_all.assign_attrs(description='Model run results for all sites.')

        ds_bay = xr.concat([df.to_xarray() for df in col_res_bay], dim="Model Runs")
        ds_bay = ds_bay.assign_attrs(description='Model run results for Bay site.')

        ds_creek = xr.concat([df.to_xarray() for df in col_res_creek], dim="Model Runs")
        ds_creek = ds_creek.assign_attrs(description='Model run results for Creek site.')

        ds_int = xr.concat([df.to_xarray() for df in col_res_interior], dim="Model Runs")
        ds_int = ds_int.assign_attrs(description='Model run results for Interior site.')

        # Export model runs    
        ds_all.to_netcdf(path=export_runs+'_wc_all_'+str(i)+'.nc', mode='w')
        ds_bay.to_netcdf(path=export_runs+'_wc_bay_'+str(i)+'.nc', mode='w')
        ds_creek.to_netcdf(path=export_runs+'_wc_creek_'+str(i)+'.nc', mode='w')
        ds_int.to_netcdf(path=export_runs+'_wc_interior_'+str(i)+'.nc', mode='w')

# number of elements
n = int(input("Enter number of model runs : "))

run_collection(n)
    
