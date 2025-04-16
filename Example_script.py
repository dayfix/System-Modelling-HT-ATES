# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 10:15:06 2025

@author: 6100430
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main2_publish import geothermal, demand_class, gas_boiler, Solar_collector, system,\
    economic_analysis, LCOE_calc_Yang, CO2_emissions_calc, system_plot
from ATES_obj_publish import ATES_obj
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    total_result = pd.DataFrame() #initialize dataframe for storing results
    
    
    timestep = 3600 #timestep in seconds
    
    #Create demand and DH components
    demand = demand_class(T_in =90,T_out = 55,example_demand="Amsterdam")
    gas = gas_boiler()
    geo = geothermal(power=6000,T_out=90)#,flow_rate = 320)
    ATES = ATES_obj([geo],max_V = 250,thickness=40,kh=5,ani=4,T_ground=15)
    # solar = Solar_collector(peak_power = 120)
    supply = [geo,ATES, gas]

    #Run calculations
    result,df_flow = system(demand,supply,len_timestep=timestep)

    #Plot results
    system_plot(result, supply, demand, len_timestep=timestep, setting ='demand_met')
    system_plot(result, supply, demand, len_timestep=timestep, setting ='ordered')
    total_result = pd.concat([total_result,result])
    
    RES_share_list = []
    boiler_list = []

           
    df_eco = economic_analysis(result, supply)
    sum_cost = 0
    
    df_CO2 = CO2_emissions_calc(result, supply)
    for i in supply:
        if np.isnan(df_eco.loc[i.name]["LCOE"]):
            pass
        else:
            sum_cost = sum_cost + df_eco.loc[i.name]["LCOE"]*sum(result[str(i.name)+" corrected"])
    for i in range(len(df_eco)):
        print("LCOH of ", df_eco.iloc[i,0]," = ", round(df_eco.iloc[i].loc["LCOE"],3),"euro/kWh")
    plt.show()