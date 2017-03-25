# -*- coding: utf-8 -*-
"""
``systems`` classes represent the controlled systems, with methods to collect 
measurements from or set control inputs to the system.  This representation 
can be real or emulated using a detailed simulation model.  A common interface
to the controlled system in both cases allows for algorithm development and
testing on a simulation with easy transition to the real system.  Measurement 
data can then be passed to ``models`` objects to estimate or validate model
parameters.  Measurement data has a specified variable organization in the form
of a Python dictionary in order to aid its use by other objects.  It is as 
follows:

.. _Key:

system.measurements = {"Measurement Variable Name" : {"Measurement Key" : mpcpy.Variables.Timeseries/Static}}.

The measurement variable name should match the variable that is measured in a 
model in the emulation case, or match the point name that is measured in a real
system case.  The measurement keys are from the following list:

- Simulated - timeseries variable for simulated measurement (yielded by ``models`` objects)
- Measured - timeseries variable for real measurement (yielded by ``systems`` objects)
- Sample - static variable for measurement sample rate
- SimulatedError - timeseries variable for simulated standard error
- MeasuredError - timeseries variable for measured standard error


=========
Emulation
=========

Emulation objects are used to simulate the performance of a real system and
collect the results of the simulation as measurements.  Models used for such 
simulations are often detailed physical models and are not necessarily the 
same as a model used for optimization.  A model for this purpose should be 
instantiated as a ``models`` object instead of a ``systems`` object.

Classes
=======

.. autoclass:: mpcpy.systems.EmulationFromFMU
    :members: collect_measurements


==== 
Real
====

Real objects are used to find and collect measurements from a real system.

Classes
=======

.. autoclass:: mpcpy.systems.RealFromCSV
    :members: collect_measurements

"""

from abc import ABCMeta, abstractmethod
from mpcpy import utility

#%% System class
class _System(utility.mpcpyPandas, utility.Building):
    '''Base class for representing systems.

    '''  

    __metaclass__ = ABCMeta;
    
    @abstractmethod
    def collect_measurements():
        '''Collect system measurements from the specified source.

        '''

        pass;

#%% System implementations
class _Emulation(_System):
    '''Base class for an emulated system class.

    '''

    __metaclass__ = ABCMeta;

    def collect_measurements(self, start_time, final_time):
        '''Simulate model and collect measurement data for the emulated system.
        
        Parameters
        ----------
        start_time : string
            Start time of measurements collection.
        final_time : string
            Final time of measurements collection.
            
        Yields
        ------
        
        measurements : dictionary
            measurements attribute.
        
        '''

        self._set_time_interval(start_time, final_time);
        self._simulate();
        for key in self.measurements.keys():
            self.measurements[key]['Measured'] = self.measurements[key]['Simulated']; 

class _Real(_System):
    '''Base class for a real system.

    '''

    __metaclass__ = ABCMeta;

    def collect_measurements(self, start_time, final_time):
        '''Search for and collect measurement data for the real system.
        
        Parameters
        ----------
        start_time : string
            Start time of measurements collection.
        final_time : string
            Final time of measurements collection.
            
        Yields
        ------
        
        measurements : dictionary
            Measurement attribute.
        
        '''

        self._set_time_interval(start_time, final_time);
        self._collect_data();

#%% Source implementations
class EmulationFromFMU(_Emulation, utility.FMU):
    '''System emulation by FMU simulation.
    
    Parameters
    ----------
    fmupath : string
        FMU file path.
    measurements : dictionary
        {"Measurement Name" : {"Sample" : mpcpy.Variables.Static}}.

    Attributes
    ----------
    measurements : dictionary
        {"Measurement Name" : {"Measurement Key_" : mpcpy.Variables.Timeseries/Static}}.
    zone_names : [strings]
        List of zone names.
    weather_data : dictionary
        ``exodata`` weather object data attribute.
    internal_data : dictionary
        ``exodata`` internal object data attribute.
    control_data : dictionary
        ``exodata`` control object data attribute.    
    other_inputs : dictionary
        ``exodata`` other inputs object data attribute.    
    parameter_data : dictionary
        ``exodata`` parameter object data attribute.    
    lat : numeric
        Latitude in degrees.  For timezone.
    lon : numeric
        Longitude in degrees.  For timezone.
    tz_name : string
        Timezone name.

    '''

    def __init__(self, fmupath, measurements, **kwargs):
        '''Constructor of system emulation by FMU.

        '''

        self.fmupath = fmupath;
        self.measurements = measurements
        self.input_names = self._get_input_names();        
        self._parse_building_kwargs(kwargs);
        self._parse_time_zone_kwargs(kwargs);
        
    def _simulate(self):
        '''Simulate the fmu.

        '''

        self._simulate_fmu();
        
class RealFromCSV(_Real, utility.DAQ):
    '''System measured data located in CSV.
    
    Parameters
    ----------
    filepath : string
        CSV file path.
    measurements : dictionary
        {"Measurement Name" : {"Sample" : mpcpy.Variables.Static}}.
    variable_map : dictionary
        {"Column Header Name" : ("Measurement Variable Name", mpcpy.Units.unit)}.        

    Attributes
    ----------
    measurements : dictionary
        {"Measurement Variable Name" : {{"Measurement Key_" : mpcpy.Variables.Timeseries/Static}}.
    zone_names : [strings]
        List of zone names.
    weather_data : dictionary
        ``exodata`` weather object data attribute.
    internal_data : dictionary
        ``exodata`` internal object data attribute.
    control_data : dictionary
        ``exodata`` control object data attribute.    
    other_inputs : dictionary
        ``exodata`` other inputs object data attribute.    
    parameter_data : dictionary
        ``exodata`` parameter object data attribute.    
    lat : numeric
        Latitude in degrees.  For timezone.
    lon : numeric
        Longitude in degrees.  For timezone.
    tz_name : string
        Timezone name.
        
    '''
    
    def __init__(self, filepath, measurements, variable_map, **kwargs):
        ''' Constructor of system measured data located in CSV.

        '''

        self.filepath = filepath;
        self.measurements = measurements;
        self.variable_map = variable_map;
        self._parse_daq_kwargs(kwargs);
        self._parse_time_zone_kwargs(kwargs);
        
    def _collect_data(self):
        '''Collect measurement data from csv.

        '''

        self._read_timeseries_from_csv();

    def _translate_variable_map(self):
        '''Translate csv column to measurement variable.

        '''

        varname = self.variable_map[self._key][0];
        unit = self.variable_map[self._key][1];
        self.measurements[varname]['Measured'] = self._dataframe_to_mpcpy_ts_variable(self._df_csv, self._key, varname, unit, \
                                                                                     start_time=self.start_time, final_time=self.final_time, \
                                                                                     cleaning_type = self._cleaning_type, \
                                                                                     cleaning_args = self._cleaning_args);             