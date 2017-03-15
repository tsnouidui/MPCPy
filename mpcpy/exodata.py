# -*- coding: utf-8 -*-
"""
``exodata`` classes are responsible for the representation of exogenous data, 
with methods to collect this data from various sources and process it for use 
within MPCPy.  This data comes from sources outside of MPCPy and are not 
measurements of the system of interest.  The data is split into categories, or 
types, in order to standardize the organization of variables within the data 
for a particular type, in the form of a python dictionary, and to allow for 
any specific data processing that may be required.  This allows exogenous data 
objects to be used throughout MPCPy regardless of their data source.  To add a 
data source, one only need to create a class that can convert the data format 
in the source to that standardized in MPCPy.

=======   
Weather
=======

Weather data represents the conditions of the ambient environment.  
Weather data objects have special methods for checking the validity of 
data and use supplied data to calculate data not directly measured, for 
example black sky temperature, wet bulb temperature, and sun position.  
Exogenous weather data has the following organization:

``weather.data = {"Weather Variable Name" : mpcpy.Variables.Timeseries}``

The weather variable names should match those input variables in the model 
and be chosen from the list found in the following list:

- weaPAtm - atmospheric pressure
- weaTDewPoi - dew point temperature
- weaTDryBul - dry bulb temperature
- weaRelHum - relative humidity
- weaNOpa - opaque sky cover
- weaCelHei - cloud height
- weaNTot - total sky cover
- weaWinSpe - wind speed
- weaWinDir - wind direction
- weaHHorIR - horizontal infrared irradiation
- weaHDirNor - direct normal irradiation
- weaHGloHor - global horizontal irradiation
- weaHDifHor - diffuse horizontal irradiation
- weaIAveHor - global horizontal illuminance
- weaIDirNor - direct normal illuminance
- weaIDifHor - diffuse horizontal illuminance
- weaZLum - zenith luminance
- weaTBlaSky - black sky temperature
- weaTWetBul - wet bulb temperature
- weaSolZen - solar zenith angle
- weaCloTim - clock time
- weaSolTim - solar time
- weaTGnd - ground temperature

Ground temperature is an exception to the data dictionary format due to 
the possibility of different temperatures at multiple depths. Therefore, 
the dictionary format for ground temperature is:

``weather.data["weaTGnd"] = {"Depth" : mpcpy.Variables.Timeseries}``
 
Classes
=======

.. autoclass:: mpcpy.exodata.WeatherFromEPW
    :members: collect_data, display_data, get_base_data

.. autoclass:: mpcpy.exodata.WeatherFromCSV
    :members: collect_data, display_data, get_base_data


========   
Internal
========

Internal data represents zone heat gains that may come from people, lights, 
or equipment.  Internal data objects have special methods for sourcing these 
heat gains from a predicted occupancy model.  Exogenous internal data has 
the following organization:

``internal.data = {"Zone Name" : {"Internal Variable Name" : mpcpy.Variables.Timeseries}}``

The internal variable names should be chosen from the following list:

- intCon - convective internal load
- intRad - radiative internal load
- intLat - latent internal load

The input names in the model should follow the convention 
``internalVariableName_zoneName``.  For example, the convective load input 
for the zone "west" should have the name ``intCon_west``.

Classes
=======

.. autoclass:: mpcpy.exodata.InternalFromCSV
    :members: collect_data, display_data, get_base_data
    
.. autoclass:: mpcpy.exodata.InternalFromOccupancyModel
    :members: collect_data, display_data, get_base_data
    

=======
Control
=======

Control data represents control inputs to a system or model.  The variables 
listed in a Control data object are special in that they are considered 
optimization variables during model optimization. Exogenous control data has 
the following organization:

``control.data = {"Control Variable Name" : mpcpy.Variables.Timeseries}``

The control variable names should match the control input variables of the model.

Classes
=======

.. autoclass:: mpcpy.exodata.ControlFromCSV
    :members: collect_data, display_data, get_base_data


===========
Other Input
===========

Other Input data represents miscellaneous inputs to a model.  The variables 
listed in an Other Inputs data object are not acted upon in any special way.  
Other input data has the following organization:

``other_input.data = {"Other Input Variable Name" : mpcpy.Variables.Timeseries}``

The other input variable names should match those of the model.

Classes
=======

.. autoclass:: mpcpy.exodata.OtherInputFromCSV
    :members: collect_data, display_data, get_base_data
    
=====
Price
=====

Price data represents price signals from utility or district energy systems 
for things such as energy consumption, demand, or other services.  Price data 
object variables are special because they are used for optimization objective 
functions involving price signals.  Exogenous price data has the following 
organization:

``price.data = {"Price Variable Name" : mpcpy.Variables.Timeseries}``

The price variable names should be chosen from the following list:

- pi_e - electrical energy price

Classes
=======

.. autoclass:: mpcpy.exodata.PriceFromCSV
    :members: collect_data, display_data, get_base_data


===========
Constraints
===========

Constraint data represents limits to which the control and state variables of 
an optimization solution must abide.  Constraint data object variables are 
included in the optimization problem formulation.  Exogenous constraint data 
has the following organization:

``constraints.data = {"State or Control Variable Name" : {"Constraint Variable Type" : mpcpy.Variables.Timeseries/Static}}``

The state or control variable name must match those that are in the model.  
The constraint variable types should be chosen from the following list:

- LTE - less than or equal to (Timeseries)
- GTE - greater than or equal to (Timeseries)
- E - equal to (Timeseries)
- Initial - initial value (Static)
- Final - final value (Static)
- Cyclic - initial value equals final value (Static - Boolean)

Classes
=======

.. autoclass:: mpcpy.exodata.ConstraintFromCSV
    :members: collect_data, display_data, get_base_data 
    
.. autoclass:: mpcpy.exodata.ConstraintFromOccupancyModel
    :members: collect_data, display_data, get_base_data     


==========
Parameters
==========

Parameter data represents inputs or coefficients of models that do not change 
with time during a simulation, which may need to be learned using system 
measurement data. Parameter data object variables are set when simulating 
models, and are estimated using model learning techniques if flagged to do so.
Exogenous parameter data has the following organization:

{"Parameter Name" : {"Parameter Variable Name" : mpcpy.Variables.Static}}

The parameter name must match that which is in the model.  The parameter 
variable names should be chosen from the following list:

- Free - boolean flag for inclusion in model learning algorithms
- Value - value of the parameter, which is also used as an initial guess for model learning algorithms
- Minimum - minimum value of the parameter for model learning algorithms
- Maximum - maximum value of the parameter for model learning algorithms
- Covariance - covariance of the parameter for model learning algorithms

Classes
=======

.. autoclass:: mpcpy.exodata.ParameterFromCSV
    :members: collect_data, display_data, get_base_data 
    
"""

from abc import ABCMeta
from mpcpy import utility
import numpy as np
import pandas as pd
from tzwhere import tzwhere
from dateutil.relativedelta import relativedelta
from pytz import exceptions as pytz_exceptions
from mpcpy import units
from mpcpy import variables
     
#%% Abstract source interface class
class _Type(utility.mpcpyPandas):
    '''Base class for exogenous data sources.'''
    __metaclass__ = ABCMeta;
    
    def collect_data(self, start_time=None, final_time=None):
        '''Collect data from specified source.
        
        Parameters
        ----------
        start_time : string
            Start time of data collection.
        final_time : string
            Final time of data collection.
            
        Yields
        ------
        
        data : dictionary
            Data attribute.
        
        '''
        
        self._collect_data(start_time, final_time);
    
    def display_data(self):
        '''Get data in display units as pandas dataframe.
        
        Returns
        -------
        
        df : ``pandas`` dataframe
            Timeseries dataframe in display units.
        
        '''
        
        self._make_mpcpy_ts_list();
        df = self._mpcpy_ts_list_to_dataframe(self._ts_list, display_data = True);
        return df;
        
    def get_base_data(self):
        '''Get data in base units as pandas dataframe.
        
        Returns
        -------
        
        df : ``pandas`` dataframe
            Timeseries dataframe in base units.
            
        '''
        
        self._make_mpcpy_ts_list();        
        df = self._mpcpy_ts_list_to_dataframe(self._ts_list, display_data = False);
        return df;
               
#%% Source implementations

## Weather       
class _Weather(_Type, utility.FMU):
    '''Mixin class for weather data.'''
        
    def _make_mpcpy_ts_list(self):
        '''Make mpcpy timeseries list.'''
        self._ts_list = [];
        for key in self.data.keys():
            if self.data[key].variability == 'Timeseries':
                self._ts_list.append(self.data[key]);        
           
    def _translate_variable_map(self):
        '''Translate csv column to data variable.'''
        varname = self.variable_map[self._key][0];
        unit = self.variable_map[self._key][1];        
        self.data[varname] = self._dataframe_to_mpcpy_ts_variable(self._df_csv, self._key, varname, unit, \
                                                                 start_time=self.start_time, final_time=self.final_time, \
                                                                 cleaning_type = self._cleaning_type, \
                                                                 cleaning_args = self._cleaning_args);
                                                                 
    def _checkCelHei(self):
        ''' Check and convert ceiling height data.
        
        See Buildings.BoundaryConditions.WeatherData.ReaderTMY3.
        
        '''
        var = self.data['weaCelHei'];
        ts_in = var.get_base_data();
        M_in = ts_in.get_values();     
        M_out = [];
        for data_point in M_in:
            if data_point > 20000:
                M_out.append(2000);
            else:
                M_out.append(data_point);
        ts_out = pd.Series(data = M_out, index = ts_in.index);
        var.set_display_unit(var.get_base_unit());
        var.set_data(ts_out);
        self.data['weaCelHei'] = var;
        
    def _checkPAtm(self):
        ''' Check and convert atmospheric pressure data.
        
        See Buildings.BoundaryConditions.WeatherData.ReaderTMY3.
        
        '''
        var = self.data['weaPAtm'];
        index = var.get_base_data().index;
        M_in = var.get_base_data().get_values();
        M_out = 101325*np.ones(len(M_in));
        ts = pd.Series(data = M_out, index = index, name = 'weaPAtm');
        var.set_display_unit(var.get_base_unit());
        var.set_data(ts);
        self.data['weaPAtm'] = var;
        
    def _checkNOpa(self):
        ''' Check and convert opaque sky data.
        
        See Buildings.BoundaryConditions.WeatherData.ReaderTMY3.
        
        '''
        var = self.data['weaNOpa'];
        ts_in = var.get_base_data();
        M_in = ts_in.get_values();     
        M_out = [];
        M_max = 1.0;
        M_min = 0.011;
        for data_point in M_in:
            if data_point > M_max:
                M_out.append(M_max);
            elif data_point < M_min:
                M_out.append(M_min);
            else:
                M_out.append(data_point);
        ts_out = pd.Series(data = M_out, index = ts_in.index);
        var.set_display_unit(var.get_base_unit());
        var.set_data(ts_out);                  
        self.data['weaNOpa'] = var;
        
    def _checkNTot(self):
        ''' Check and convert total sky coverage data.
        
        See Buildings.BoundaryConditions.WeatherData.ReaderTMY3.
        
        '''
        var = self.data['weaNTot'];
        ts_in = var.get_base_data();
        M_in = ts_in.get_values();  
        M_out = [];
        M_max = 1.0;
        M_min = 0.011;
        for data_point in M_in:
            if data_point > M_max:
                M_out.append(M_max);
            elif data_point < M_min:
                M_out.append(M_min);
            else:
                M_out.append(data_point);
        ts_out = pd.Series(data = M_out, index = ts_in.index);
        var.set_display_unit(var.get_base_unit());
        var.set_data(ts_out);   
        self.data['weaNTot'] = var;
        
    def _checkRelHum(self):
        ''' Check and convert relative humidity data.
        
        See Buildings.BoundaryConditions.WeatherData.ReaderTMY3.
        
        '''
        var = self.data['weaRelHum'];
        ts_in = var.get_base_data();
        M_in = ts_in.get_values();
        M_out = [];
        M_max = 0.989;
        M_min = 0.0;
        for data_point in M_in:
            if data_point > M_max:
                M_out.append(M_max);
            elif data_point < M_min:
                M_out.append(M_min);
            else:
                M_out.append(data_point);
        ts_out = pd.Series(data = M_out, index = ts_in.index);
        var.set_display_unit(var.get_base_unit());
        var.set_data(ts_out); 
        self.data['weaRelHum'] = var;
        
    def _process_weather_data(self):
        '''Use process weather fmu to calculate other necessary weather data.'''
        
        # Set filepath for fmu
        weatherdir = utility.get_MPCPy_path() + '/resources/weather';
        fmuname = 'RapidMPC_BoundaryConditions_WeatherProcessor.fmu';
        self.fmupath = weatherdir+'/'+fmuname;
        # Set parameters for fmu
        self.parameter_data = {};
        self.parameter_data['lat'] = {};
        self.parameter_data['lat']['Value'] = self.lat;
        self.parameter_data['lon'] = {};
        self.parameter_data['lon']['Value'] = self.lon;
        self.parameter_data['timZon'] = {};        
        self.parameter_data['timZon']['Value'] = self.time_zone;
        self.parameter_data['modTimOffset'] = {};   
        self.parameter_data['modTimOffset']['Value'] = variables.Static('modTimeOffest', self.year_start_seconds, units.s);
        for key in self.parameter_data.keys():
            self.parameter_data[key]['Free'] = variables.Static(key+'_free', False, units.boolean);
        # Set measurements for fmu
        self.measurements = {};
        for key in self.process_variables:
            self.measurements[key] = {};
            self.measurements[key]['Sample'] = variables.Static(key+'_Sample', 3600, units.s);
        # Simulate the fmu
        self._simulate_fmu();
        # Add process var data 
        for key in self.process_variables:
            self.data[key] = self.measurements[key]['Simulated'];

    def _create_input_mpcpy_ts_list_sim(self):
        # Set input_object for fmu
        self._input_mpcpy_ts_list = (self.data['weaPAtm'], self.data['weaTDewPoi'], \
                                     self.data['weaTDryBul'], self.data['weaRelHum'], \
                                     self.data['weaNOpa'], self.data['weaCelHei'], \
                                     self.data['weaNTot'], self.data['weaWinSpe'], \
                                     self.data['weaWinDir'], self.data['weaHHorIR'], \
                                     self.data['weaHDirNor'], self.data['weaHGloHor']);        
        
## Internal       
class _Internal(_Type):
    '''Mixin class for internal data.'''
        
    def _make_mpcpy_ts_list(self):
        '''Make mpcpy timeseries list.'''        
        self._ts_list = [];
        for zone in self.data.keys():
            for key in self.data[zone].keys():
                if self.data[zone][key].variability == 'Timeseries':
                    self._ts_list.append(self.data[zone][key]);
        
    def _translate_variable_map(self):
        '''Translate csv column to data variable.'''
        zone = self.variable_map[self._key][0];
        load = self.variable_map[self._key][1];
        varname = load + '_' + zone;
        unit = self.variable_map[self._key][2];        
        try:
            self.data[zone][load] = self._dataframe_to_mpcpy_ts_variable(self._df_csv, self._key, varname, unit, \
                                                                       start_time=self.start_time, final_time=self.final_time, \
                                                                       cleaning_type = self._cleaning_type, \
                                                                       cleaning_args = self._cleaning_args);
        except KeyError:
            self.data[zone] = {};
            self.data[zone][load] = self._dataframe_to_mpcpy_ts_variable(self._df_csv, self._key, varname, unit, \
                                                                       start_time=self.start_time, final_time=self.final_time, \
                                                                       cleaning_type = self._cleaning_type, \
                                                                       cleaning_args = self._cleaning_args);        

         
## Controls       
class _Control(_Type):
    '''Mixin class for control data.'''
        
    def _make_mpcpy_ts_list(self):
        '''Make mpcpy timeseries list.'''
        self._ts_list = [];
        for key in self.data.keys():
            if self.data[key].variability == 'Timeseries':
                self._ts_list.append(self.data[key]);        
           
    def _translate_variable_map(self):
        '''Translate csv column to data variable.'''
        varname = self.variable_map[self._key][0];
        unit = self.variable_map[self._key][1];        
        self.data[varname] = self._dataframe_to_mpcpy_ts_variable(self._df_csv, self._key, varname, unit, \
                                                                 start_time=self.start_time, final_time=self.final_time, \
                                                                 cleaning_type = self._cleaning_type, \
                                                                 cleaning_args = self._cleaning_args);   
                                                                 
## Other_Inputs       
class _OtherInput(_Type):
    '''Mixin class for other input data.'''
       
    def _make_mpcpy_ts_list(self):
        '''Make mpcpy timeseries list.'''
        self._ts_list = [];
        for key in self.data.keys():
            if self.data[key].variability == 'Timeseries':
                self._ts_list.append(self.data[key]);        
           
    def _translate_variable_map(self):
        '''Translate csv column to data variable.'''
        varname = self.variable_map[self._key][0];
        unit = self.variable_map[self._key][1];        
        self.data[varname] = self._dataframe_to_mpcpy_ts_variable(self._df_csv, self._key, varname, unit, \
                                                                 start_time=self.start_time, final_time=self.final_time, \
                                                                 cleaning_type = self._cleaning_type, \
                                                                 cleaning_args = self._cleaning_args);
                                                                 
## Constraints       
class _Constraint(_Type):
    '''Mixin class for constraint data.'''
        
    def _make_mpcpy_ts_list(self):
        '''Make mpcpy timeseries list.'''
        self._ts_list = [];
        for state in self.data.keys():
            for key in self.data[state].keys():
                if self.data[state][key].variability == 'Timeseries':
                    self._ts_list.append(self.data[state][key]);
        
    def _translate_variable_map(self):
        '''Translate csv column to data variable.'''
        state = self.variable_map[self._key][0];
        key = self.variable_map[self._key][1];
        varname = state + '_' + key;
        unit = self.variable_map[self._key][2];        
        try:
            self.data[state][key] = self._dataframe_to_mpcpy_ts_variable(self._df_csv, self._key, varname, unit, \
                                                                       start_time=self.start_time, final_time=self.final_time, \
                                                                       cleaning_type = self._cleaning_type, \
                                                                       cleaning_args = self._cleaning_args);
        except KeyError:
            self.data[state] = {};
            self.data[state][key] = self._dataframe_to_mpcpy_ts_variable(self._df_csv, self._key, varname, unit, \
                                                                       start_time=self.start_time, final_time=self.final_time, \
                                                                       cleaning_type = self._cleaning_type, \
                                                                       cleaning_args = self._cleaning_args);


## Prices       
class _Price(_Type):
    '''Mixin class for price data.'''
        
    def _make_mpcpy_ts_list(self):
        '''Make mpcpy timeseries list.'''
        self._ts_list = [];
        for key in self.data.keys():
            if self.data[key].variability == 'Timeseries':
                self._ts_list.append(self.data[key]);

    def _translate_variable_map(self):
        '''Translate csv column to data variable.'''
        varname = self.variable_map[self._key][0];
        unit = self.variable_map[self._key][1];
        self.data[varname] = self._dataframe_to_mpcpy_ts_variable(self._df_csv, self._key, varname, unit, \
                                                                 start_time=self.start_time, final_time=self.final_time, \
                                                                 cleaning_type = self._cleaning_type, \
                                                                 cleaning_args = self._cleaning_args);


## Parameters       
class _Parameter(_Type):
    '''Mixin class for parameter data.'''

    def display_data(self):
        '''Display data as pandas dataframe.

        Returns
        -------

        df : ``pandas`` dataframe
            Dataframe in display units.
            
        '''

        d = {};
        for key in self.data.keys():
            d[key] = {};
            for subkey in self.data[key].keys():
                d[key][subkey] = self.data[key][subkey].display_data();
                if subkey == 'Value':
                    d[key]['Unit'] = self.data[key][subkey].get_display_unit_name();
        df_coefficients = pd.DataFrame(data = d).transpose();
        df_coefficients.index.name = 'Name';
        return df_coefficients;

    def get_base_data(self):
        '''Get base data as pandas dataframe.

        Returns
        -------

        df : ``pandas`` dataframe
            Dataframe in base units.

        '''

        d = {};
        for key in self.data.keys():
            d[key] = {};
            for subkey in self.data[key].keys():
                d[key][subkey] = self.data[key][subkey].get_base_data();
        df_coefficients = pd.DataFrame(data = d);
        return df_coefficients;
   
#%% Weather source implementations    
class WeatherFromEPW(_Weather):
    '''Collects weather data from an EPW file.
    
    Parameters
    ----------
    filepath :  string
        EPW file path.

    Attributes
    ----------
    data : dictionary
        {"Weather Variable Name" : mpcpy.Variables.Timeseries}.
    lat : numeric
        Latitude in degrees.
    lon : numeric
        Longitude in degrees.
        
    '''
    def __init__(self, filepath):
        '''Constructor of EPW file weather data object.'''
        self.filepath = filepath;
        self._read_lat_lon_timZon_from_epw();
        self.tz = tzwhere.tzwhere();
        self.tz_name = self.tz.tzNameAt(self.lat.display_data(), self.lon.display_data());        
        self.data = {};
        self.process_variables = ['weaTBlaSky', \
                                  'weaTWetBul', \
                                  'weaHDifHor', \
                                  'weaCloTim', \
                                  'weaSolTim', \
                                  'weaSolZen'];         

    def _collect_data(self, start_time, final_time):
        '''Collect weather data from EPW file.'''
        # Set time interval
        self._set_time_interval(start_time, final_time);
        # Get bulk timeseries weather data
        self._read_timeseries_from_epw();
        # Process weather data
        self._process_weather_data();
        
        return self.data
        
    def _read_lat_lon_timZon_from_epw(self):
        '''Get Latitude, Longitude, and Time Zone from EPW file.'''
        df_epw = pd.read_csv(self.filepath, nrows = 1, header = None, usecols = [6,7,8], names = ['Latitude', 'Longitude', 'TimeZone']);
        self.lat = variables.Static('lat', df_epw.loc[0,'Latitude'], units.deg);
        self.lon = variables.Static('lon', df_epw.loc[0,'Longitude'], units.deg); 
        self.time_zone = variables.Static('timZon', df_epw.loc[0,'TimeZone'], units.hour);
        
    def _read_timeseries_from_epw(self):
        '''Read in timeseries weather data from EPW file.'''
        # Define column headers to read in from epw
        header = ['Year', 'Month', 'Day', 'Hour', 'Second', 'Unknown', \
                  'Dry bulb temperature', 'Dew point temperature', \
                  'Relative humidity', 'Atmospheric station pressure', \
                  'Extraterrestrial horizontal radiation', 'Extraterrestrial direct normal radiation', \
                  'Horizontal infrared radiation', 'Global horizontal radiation', \
                  'Direct normal radiation', 'Diffuse horizontal radiation', \
                  'Averaged global horizontal illuminance', 'Direct normal illuminance', \
                  'Diffuse horizontal illuminance', 'Zenith luminance', \
                  'Wind direction', 'Wind speed', \
                  'Total sky cover', 'Opaque sky cover', \
                  'Visibility', 'Ceiling', \
                  'Present weather observation', 'Present weather codes', \
                  'Precipitable water', 'Aerosol optical depth', \
                  'Snow depth', 'Days since last snowfall', \
                  'Albedo', 'Liquid precipitation depth', \
                  'Liquid precipitation quantity'];
        # Read in data
        df_epw = pd.read_csv(self.filepath, skiprows = 8, header = None, names=header);
        # Convert time columns to timestamp and set as index                           
        df_epw['Hour'] = df_epw['Hour'] - 1;
        
        df_epw['Time'] = str(self.start_time.year) + ' ' + df_epw['Month'].apply(str) + ' ' + df_epw['Day'].apply(str) + ' ' + df_epw['Hour'].apply(str) + ':00';
        time = pd.to_datetime(df_epw['Time'], format= '%Y %m %d %H:%M');
        df_epw.set_index(time, inplace=True);
        # Remove unneeded columns
        df_epw = df_epw.drop(['Time', 'Year', 'Month', 'Day', 'Hour', 'Second', 'Unknown'], axis = 1);
        #  Perform data swap for epw (see Buildings.BoundaryConditions.WeatherData.ReaderTMY3 info)  
        df_epw_last_row = df_epw.head(1);
        df_epw = df_epw_last_row.append(df_epw.iloc[:-1], ignore_index=False);
        new_index = df_epw.index[0:1].append(df_epw.index[1:] + pd.DateOffset(hours=1));
        df_epw.set_index(new_index, inplace=True);
        # Treat daylight savings time
        try:
            df_epw = df_epw.tz_localize(self.tz_name);
        except pytz_exceptions.NonExistentTimeError as time_nonexist:
            time_nonexist = pd.to_datetime(time_nonexist.args[0])
            if time_nonexist.month == 3:
                df_epw_st = df_epw[df_epw.index < time_nonexist];
                df_epw_dst = df_epw[df_epw.index >= time_nonexist];
                df_epw_dst = df_epw_dst.shift(periods = 1, freq = 'H');
                df_epw = pd.concat([df_epw_st, df_epw_dst], axis = 0);
        try:
            df_epw = df_epw.tz_localize(self.tz_name);
        except pytz_exceptions.AmbiguousTimeError as time_ambiguous:
            time_ambiguous = pd.to_datetime(time_ambiguous.args[0].split("'")[1])
            if time_ambiguous.month == 11:
                df_epw_dst = df_epw[df_epw.index < time_ambiguous].tz_localize(self.tz_name);
                df_epw_st = df_epw[(df_epw.index > time_ambiguous + relativedelta(hours = 1))].shift(periods = -1, freq = 'H').tz_localize(self.tz_name);
                df_epw_amb = df_epw[(df_epw.index >= time_ambiguous) & (df_epw.index <= time_ambiguous + relativedelta(hours = 1))];
                df_epw_amb_0 = df_epw_amb.iloc[0:1].tz_localize(self.tz_name, ambiguous = np.array([True]));
                df_epw_amb_1 = df_epw_amb.iloc[1:2].shift(periods = -1, freq = 'H').tz_localize(self.tz_name, ambiguous = np.array([False]));
                df_epw = pd.concat([df_epw_dst, df_epw_amb_0, df_epw_amb_1, df_epw_st], axis = 0);
        #  Retrieve data (not all is retrieved)
        for key in header:
            # Convert to mpcpy standard
            if key == 'Atmospheric station pressure':
                varname = 'weaPAtm';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.Pa, start_time = self.start_time, final_time = self.final_time);
                self._checkPAtm();
            elif key == 'Dew point temperature':
                varname = 'weaTDewPoi';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.degC, start_time = self.start_time, final_time = self.final_time);
            elif key == 'Dry bulb temperature':
                varname = 'weaTDryBul';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.degC, start_time = self.start_time, final_time = self.final_time);
            elif key == 'Relative humidity':
                varname = 'weaRelHum';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.percent, start_time = self.start_time, final_time = self.final_time);                
                self._checkRelHum();
            elif key == 'Opaque sky cover':
                varname = 'weaNOpa';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.unit10, start_time = self.start_time, final_time = self.final_time);                
                self._checkNOpa();
            elif key == 'Ceiling':
                varname = 'weaCelHei';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.m, start_time = self.start_time, final_time = self.final_time);                
                self._checkCelHei();
            elif key == 'Total sky cover':
                varname = 'weaNTot';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.unit10, start_time = self.start_time, final_time = self.final_time);                 
                self._checkNTot();
            elif key == 'Wind speed':
                varname = 'weaWinSpe';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.m_s, start_time = self.start_time, final_time = self.final_time);
            elif key == 'Wind direction':
                varname = 'weaWinDir';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.deg, start_time = self.start_time, final_time = self.final_time);
            elif key == 'Horizontal infrared radiation':
                varname = 'weaHHorIR';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.W_m2, start_time = self.start_time, final_time = self.final_time);
            elif key == 'Direct normal radiation':
                varname = 'weaHDirNor';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.W_m2, start_time = self.start_time, final_time = self.final_time); 
            elif key == 'Global horizontal radiation':
                varname = 'weaHGloHor';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.W_m2, start_time = self.start_time, final_time = self.final_time);
            elif key == 'Diffuse horizontal radiation':
                varname = 'weaHDifHor';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.W_m2, start_time = self.start_time, final_time = self.final_time);
            elif key == 'Averaged global horizontal illuminance':
                varname = 'weaIAveHor';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.lx, start_time = self.start_time, final_time = self.final_time);
            elif key == 'Direct normal illuminance':
                varname = 'weaIDirNor';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.lx, start_time = self.start_time, final_time = self.final_time);
            elif key == 'Diffuse horizontal illuminance':
                varname = 'weaIDifHor';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.lx, start_time = self.start_time, final_time = self.final_time);
            elif key == 'Zenith luminance':
                varname = 'weaZLum';
                self.data[varname] = self._dataframe_to_mpcpy_ts_variable(df_epw, key, varname, units.cd_m2, start_time = self.start_time, final_time = self.final_time);          
        # Time shift the solar data back 30 minutes by linear interpolation (see Buildings.BoundaryConditions.WeatherData.ReaderTMY3 info)
        for key in self.data.keys():
            if key in ['weaHHorIR', 'weaHGloHor', 'weaHDirNor', 'weaHDifHor', \
                     'weaIAveHor', 'weaIDirNor', 'weaIDifHor', 'weaZLum']:
                ts_old = self.data[key].display_data();
                ts = ts_old.resample('30T').interpolate(method='time');
                ts = ts.shift(freq = '-30T');
                ts = ts.resample(rule='H', how = 'first');
                ts = ts.ix[1:].append(ts_old.tail(n=1));
                self.data[key].set_data(ts);
                     
class WeatherFromCSV(_Weather, utility.DAQ):
    '''
    Collects weather data from a CSV file.

    Parameters
    ----------
    filepath : string
        CSV file path.
    variable_map : dictionary
        {"Column Header Name" : ("Weather Variable Name", mpcpy.Units.unit)}.

    Attributes
    ----------
    data : dictionary
        {"Weather Variable Name" : mpcpy.Variables.Timeseries}.
    lat : numeric
        Latitude in degrees.
    lon : numeric
        Longitude in degrees.
    tz_name : string
        Timezone name.        

    '''
    def __init__(self, filepath, variable_map, **kwargs):
        '''Constructor of CSV file weather data object.'''
        self.filepath = filepath;  
        self.data = {};   
        self.variable_map = variable_map;          
        # Process Variables
        if 'process_variables' in kwargs:
            self.process_variables = kwargs['process_variables'];
        else:
            self.process_variables = None;
        # Common kwargs
        self._parse_daq_kwargs(kwargs);
        self._parse_time_zone_kwargs(kwargs);
        # Assert geography
        assert(bool(self.lat) == True);
        assert(bool(self.lon) == True);
           
    def _collect_data(self, start_time, final_time):
        '''Collect weather data from CSV file.'''
        # Set time interval
        self._set_time_interval(start_time, final_time);
        # Get bulk time series        
        self._read_timeseries_from_csv();
        # Process weather data
        if self.process_variables is not None:
            self._process_weather_data();   
                                             
#%% Internal source implementations
class InternalFromCSV(_Internal, utility.DAQ):
    '''
    Collects internal data from a CSV file.

    Parameters
    ----------
    filepath : string
        CSV file path.
    variable_map : dictionary
        {"Column Header Name" : ("Zone Name", "Internal Variable Name", mpcpy.Units.unit)}.

    Attributes
    ----------
    data : dictionary
        {"Zone Name" : {"Internal Variable Name" : mpcpy.Variables.Timeseries}}.
    lat : numeric
        Latitude in degrees.  For timezone.
    lon : numeric
        Longitude in degrees.  For timezone.
    tz_name : string
        Timezone name.        

    '''
    
    def __init__(self, filepath, variable_map, **kwargs):
        ''' Constructor of csv internal source.'''
        self.filepath = filepath;
        self.data = {};
        self.variable_map = variable_map;
        # Common kwargs
        self._parse_daq_kwargs(kwargs);
        self._parse_time_zone_kwargs(kwargs);
                   
    def _collect_data(self, start_time, final_time):
        '''Collect internal data from CSV file.'''
        # Set time interval
        self._set_time_interval(start_time, final_time);
        # Get bulk time series        
        self._read_timeseries_from_csv();
        
class InternalFromOccupancyModel(_Internal):
    '''
    Collects internal data from an occupancy model.

    Parameters
    ----------
    zone_list : [string]
        List of zones.
    load_list : [[numeric, numeric, numeric]]
        List of load per person lists for [convective, radiative, latent] corresponding to zone_list.
    unit : mpcpy.Units.unit
        Unit of loads.
    occupancy_model_list : [mpcpy.Models.Occupancy]
        List of occupancy model objects corresponding to zone_list.
    

    Attributes
    ----------
    data : dictionary
        {"Zone Name" : {"Internal Variable Name" : mpcpy.Variables.Timeseries}}.
    lat : numeric
        Latitude in degrees.  For timezone.
    lon : numeric
        Longitude in degrees.  For timezone.
    tz_name : string
        Timezone name.        

    '''
    
    def __init__(self, zone_list, load_list, unit, occupancy_model_list, **kwargs):
        '''Constructor of occupancy model internal source.'''
        self.zone_list = zone_list;
        self.load_list = load_list;
        self.unit = unit;        
        self.occupancy_model_list = occupancy_model_list;
        self.data = {};
        # Common kwargs    
        self._parse_time_zone_kwargs(kwargs);
        
    def _collect_data(self, start_time, final_time):
        '''Collect internal data from an occupancy model object.'''
        # Set time interval
        self._set_time_interval(start_time, final_time);
        # Get bulk time series
        for zone, loads, occupancy_model in zip(self.zone_list, self.load_list, self.occupancy_model_list):
            self.data[zone] = {};
            for varname, load in zip(['intCon', 'intRad', 'intLat'], loads):
                ts = occupancy_model.generate_load(load);
                self.data[zone][varname] = variables.Timeseries(varname+'_'+zone, ts[self.start_time:self.final_time], self.unit);
        
#%% Control source implementations        
class ControlFromCSV(_Control, utility.DAQ):
    '''
    Collects control data from a CSV file.

    Parameters
    ----------
    filepath : string
        CSV file path.
    variable_map : dictionary
        {"Column Header Name" : ("Control Variable Name", mpcpy.Units.unit)}.

    Attributes
    ----------
    data : dictionary
        {"Control Variable Name" : mpcpy.Variables.Timeseries}.
    lat : numeric
        Latitude in degrees.  For timezone.
    lon : numeric
        Longitude in degrees.  For timezone.
    tz_name : string
        Timezone name.        

    '''
    
    def __init__(self, filepath, variable_map, **kwargs):
        ''' Constructor of csv control source.'''
        self.filepath = filepath;
        self.data = {};
        self.variable_map = variable_map;
        # Common kwargs
        self._parse_daq_kwargs(kwargs);
        self._parse_time_zone_kwargs(kwargs);             
                   
    def _collect_data(self, start_time, final_time):
        '''Collect control data from CSV file.'''
        # Set time interval
        self._set_time_interval(start_time, final_time);
        # Get bulk time series        
        self._read_timeseries_from_csv();
        
#%% Other input source implementations        
class OtherInputFromCSV(_OtherInput, utility.DAQ):
    '''
    Collects other input data from a CSV file.

    Parameters
    ----------
    filepath : string
        CSV file path.
    variable_map : dictionary
        {"Column Header Name" : ("Other Input Variable Name", mpcpy.Units.unit)}.

    Attributes
    ----------
    data : dictionary
        {"Other Input Variable Name" : mpcpy.Variables.Timeseries}.
    lat : numeric
        Latitude in degrees.  For timezone.
    lon : numeric
        Longitude in degrees.  For timezone.
    tz_name : string
        Timezone name.
    
    '''
    
    def __init__(self, csv_filepath, variable_map, **kwargs):
        ''' Constructor of csv other input source.'''
        self.name = 'otherinput_from_csv';
        self.filepath = csv_filepath;
        self.data = {};   
        # Dictionary of format {'csvHeader' : ('otherinputVarName', mpcpyUnit)}
        self.variable_map = variable_map;
        # Common kwargs
        self._parse_daq_kwargs(kwargs);
        self._parse_time_zone_kwargs(kwargs);             
                   
    def _collect_data(self, start_time, final_time):
        '''Collect other input data from CSV file.'''
        # Set time interval
        self._set_time_interval(start_time, final_time);
        # Get bulk time series        
        self._read_timeseries_from_csv();        
        
           
#%% Constraint source implementations
class ConstraintFromCSV(_Constraint, utility.DAQ):
    '''
    Collects constraint data from a CSV file.

    Parameters
    ----------
    filepath : string
        CSV file path.
    variable_map : dictionary
        {"State or Control Variable Name" : {"Constraint Variable Name" : mpcpy.Variables.Timeseries/Static}}.

    Attributes
    ----------
    data : dictionary
        {"Column Header Name" : ("State or Control Variable Name", "Constraint Variable Type", mpcpy.Units.unit)}.
    lat : numeric
        Latitude in degrees.  For timezone.
    lon : numeric
        Longitude in degrees.  For timezone.
    tz_name : string
        Timezone name.
    
    '''
    
    def __init__(self, filepath, variable_map, **kwargs):
        ''' Constructor of csv constraint source.'''
        self.filepath = filepath;
        self.data = {};
        self.variable_map = variable_map;
        # Common kwargs
        self._parse_daq_kwargs(kwargs);
        self._parse_time_zone_kwargs(kwargs);
            
    def _collect_data(self, start_time, final_time):
        '''Collect constraint data from CSV file.'''
        # Set time interval
        self._set_time_interval(start_time, final_time);
        # Get bulk time series        
        self._read_timeseries_from_csv();
        
class ConstraintFromOccupancyModel(_Constraint):
    '''
    Collects constraint data from an occupancy model.

    Parameters
    ----------
    state_variable_list : [string]
        List of variable names to be constrained.  States with multiple constraints should be listed once for each constraint type.
    values_list : [[numeric or boolean, numeric or boolean]]
        List of values for [Occupied, Unoccupied] corresponding to state_variable_list.
    constraint_type_list : [string]
        List of contraint variable types corresponding to state_variable_list. 
    unit_list : [mpcpy.Units.unit]
        List of units corresponding to each contraint type in constraint_type_list.
    occupancy_model : mpcpy.Models.Occupancy
        Occupancy model object to use.   
    

    Attributes
    ----------
    data : dictionary
        {"State or Control Variable Name" : {"Constraint Variable Type" : mpcpy.Variables.Timeseries/Static}}.
    lat : numeric
        Latitude in degrees.  For timezone.
    lon : numeric
        Longitude in degrees.  For timezone.
    tz_name : string
        Timezone name.        

    '''
    def __init__(self, state_variable_list, values_list, constraint_type_list, unit_list, occupancy_model, **kwargs):
        '''Constructor of occupancy model constraint source.'''
        self.state_variable_list = state_variable_list;
        self.values_list = values_list;
        self.constraint_type_list = constraint_type_list;
        self.unit_list = unit_list;
        self.occupancy_model = occupancy_model;
        self.data = {};        
        # Common kwargs
        self._parse_time_zone_kwargs(kwargs);
        
    def _collect_data(self, start_time, final_time):
        '''Collect constraint data from occupancy model.'''
        # Set time interval
        self._set_time_interval(start_time, final_time);
        # Get bulk time series
        for state_variable, values, constraint_type, unit in zip(self.state_variable_list, self.values_list, self.constraint_type_list, self.unit_list):
            if state_variable not in self.data:
                self.data[state_variable] = {};
            ts = self.occupancy_model.generate_constraint(values[0], values[1]);
            self.data[state_variable][constraint_type] = variables.Timeseries(state_variable+'_'+constraint_type, ts[self.start_time:self.final_time], unit);

#%% Price source implementations
class PriceFromCSV(_Price, utility.DAQ):
    '''Collects price data from a CSV file.

    Parameters
    ----------
    filepath : string
        CSV file path.
    variable_map : dictionary
        {"Column Header Name" : ("Price Variable Name", mpcpy.Units.unit)}.

    Attributes
    ----------
    data : dictionary
        {"Price Variable Name" : mpcpy.Variables.Timeseries}.
    lat : numeric
        Latitude in degrees.  For timezone.
    lon : numeric
        Longitude in degrees.  For timezone.
    tz_name : string
        Timezone name.
    
    '''        

    def __init__(self, filepath, variable_map, **kwargs):
        ''' Constructor of csv price source.'''
        self.name = 'constraint_from_csv';
        self.filepath = filepath;
        self.data = {};   
        self.variable_map = variable_map;
        # Common kwargs
        self._parse_daq_kwargs(kwargs);
        self._parse_time_zone_kwargs(kwargs);
            
    def _collect_data(self, start_time, final_time):
        '''Collect price data from a csv file.'''
        # Set time interval
        self._set_time_interval(start_time, final_time);
        # Get bulk time series        
        self._read_timeseries_from_csv();


#%% Parameter source implementations 
class ParameterFromCSV(_Parameter, utility.DAQ):
    '''
    Collects parameter data from a CSV file. The CSV file rows must be named 
    as the parameter names and the columns must be named as the parameter 
    variable names.

    Parameters
    ----------
    filepath : string
        CSV file path.

    Attributes
    ----------
    data : dictionary
        {"Parameter Name" : {"Parameter Variable Name" : mpcpy.Variables.Static}}.
    
    '''
    def __init__(self, filepath):
        ''' Constructor of csv parameter data source.'''
        self.filepath = filepath;
        self.data = {};
    def collect_data(self):
        '''Collect parameter data from CSV file.
        
        Yields
        ------
        
        data : dictionary
            Data attribute.
            
        '''
        # Read coefficients file
        df = pd.read_csv(self.filepath, index_col='Name', dtype={'Unit':str});
        # Create coefficient dictionary
        for key in df.index.values:
            self.data[key] = {};
            unit = utility.get_unit_class_from_unit_string(df.loc[key, 'Unit']);
            if df.loc[key, 'Free']:  
                self.data[key]['Free'] = variables.Static(key+'_free', True, units.boolean);
                self.data[key]['Value'] = variables.Static(key+'_val', df.loc[key, 'Value'], unit);
                self.data[key]['Minimum'] = variables.Static(key+'_min', df.loc[key, 'Minimum'], unit);
                self.data[key]['Maximum'] = variables.Static(key+'_max', df.loc[key, 'Maximum'], unit);
                self.data[key]['Covariance'] = variables.Static(key+'_cov', df.loc[key, 'Covariance'], unit);
            else: 
                self.data[key]['Free'] = variables.Static(key+'_free', False, units.boolean);
                self.data[key]['Value'] = variables.Static(key+'_val', df.loc[key, 'Value'], unit);             
