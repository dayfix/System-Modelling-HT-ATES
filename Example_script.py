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
    demand = demand_class(T_in =75,T_out = 55,example_demand="Amsterdam")
    gas = gas_boiler()
    geo = geothermal(power=5000,T_out=75)#,flow_rate = 320)
    ATES = ATES_obj([geo],max_V = 300,thickness=40,kh=5,ani=4,T_ground=15)
    # solar = Solar_collector(peak_power = 120)
    
    #Put the different suppliers in a supply list. Preferably in the following order for plotting:
        #1. Sustainable suppliers (Geo or solar)
        #2. STES (HT-ATES)
        #3. back-up unit (gas)
        
    supply = [geo,ATES, gas]

    #Run calculations, all in this system function
    result,df_flow = system(demand,supply,len_timestep=timestep)

    #Plot results
    system_plot(result, supply, demand, len_timestep=timestep, setting ='demand_met')
    system_plot(result, supply, demand, len_timestep=timestep, setting ='ordered')
    total_result = pd.concat([total_result,result])
    
    #Calculate the LCOH of components and the system.
    df_eco = economic_analysis(result, supply,incorporate_CO2=True)
    LCOE_Yang = LCOE_calc_Yang(result,supply,df_eco,lifetime_system=60)
    
    #Calculate CO2_emissions
    df_CO2 = CO2_emissions_calc(result, supply)

    #Print LCOH values of used suppliers
    for i in range(len(df_eco)):
        print("LCOH of ", df_eco.iloc[i,0]," = ", round(df_eco.iloc[i].loc["LCOE"],3),"euro/kWh")
    
    
    #%% Second example
    supply = [geo,gas]
    result,df_flow = system(demand,supply,len_timestep=timestep)
    df_eco_2 = economic_analysis(result, supply,incorporate_CO2=True)
    LCOE_Yang = LCOE_calc_Yang(result,supply,df_eco_2,lifetime_system=60)

    df_CO2_2 = CO2_emissions_calc(result, supply)

    #Calculate CAC
    df_eco.loc[:,"CO2_per_kWh"]=df_CO2.loc[:,"CO2_emission [kg]"].sum()/ sum(demand.data)
    df_eco_2.loc[:,"CO2_per_kWh"]=df_CO2_2.loc[:,"CO2_emission [kg]"].sum()/ sum(demand.data)
    #Calculate cost change and LCOE change to calculate and print CAC/CRC
    CO2_change = df_eco["CO2_per_kWh"].reset_index(drop=True) -df_eco_2["CO2_per_kWh"].reset_index(drop=True)
    Cost_change = df_eco['LCOE_System'].reset_index(drop=True) -df_eco_2['LCOE_System'].reset_index(drop=True)
    CAC = -Cost_change/CO2_change
    print("CAC of HT-ATES = ",CAC.iloc[0]," euro/kgCO2")

           
    #show plots
    plt.show()