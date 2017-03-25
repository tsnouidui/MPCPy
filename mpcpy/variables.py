# -*- coding: utf-8 -*-
"""
``variables`` classes together with ``units`` classes form the fundamental 
building blocks of data management in MPCPy.  They provide functionality for 
assigning and converting between units as well as processing timeseries data.  

Generally speaking, variables in MPCPy contain three components:

    name

        A descriptor of the variable.

    data

        Constant value or a timeseries.

    unit

        Assigned to variables and act on the data depending on the requested 
        functionality, such as converting between between units or extracting 
        the data.

A unit assigned to a variable is called the display unit and is associated 
with a quantity.  For each quantity, there is a predefined base unit.  The 
data entered into a variable with a display unit is automatically converted 
to and stored as the quantity base unit.  This way, if the display unit were 
to be changed, the data only needs to be converted to the new unit upon 
extraction.  For example, the unit Degrees Celsius is of the quantity 
temperature, for which the base unit is Kelvin.  Therefore, data entered with 
a display unit of Degrees Celsius would be converted to and stored in Kelvin.  
If the display unit were to be changed to Degrees Fahrenheit, then the data 
would be converted from Kelvin upon extraction.

Instantiation
=============

Variables are instantiated by defining the variable type and the three 
components listed in the previous section.  If the data of the variable 
does not change with time, the variable must be instantiated using the 
``variables.Static`` class.  Data supplied to static variable may be a 
single value, a list, or a numpy array.  If the data of the variable is a 
timeseries, the variable must be instantiated using the 
``variables.Timeseries`` class.  Data supplied to a timeseries variable must 
be in the form of a pandas series object with a datetime index.  This brings 
to MPCPy all of the functionality of the pandas package.  The unit assigned is 
a class chosen from the ``units`` module.

::

    # Instantiate a static variable with units in Degrees Celsius
    var = variables.Static('var', 20, units.degC)

Timeseries variables have capabilities to manage the the timezone of the data 
as well as clean the data upon instantiation with the following optional 
keyword arguments:

    tz_name

        The name of the timezone as defined by ``tzwhere``.  By default, the 
        UTC timezone is assigned to the data.  If a different timezone is 
        assigned, the data is converted to a stored in UTC.  Similar to the 
        treatment of data units, the timezone is only converted to the 
        assigned timezone upon data extraction.

    geography

        Tuple containing (latitude,longitude) in degrees.  If geography is 
        defined, the timezone associated with that location will be assigned 
        to the variable.

    cleaning_type

        The type of cleaning to be performed on the data.  This should be a 
        class selected from ``variables.Timeseries``.

    cleaning_args

        Arguments of the cleaning_type defined.


Variable Management
===================

Accessing Data
--------------
Data may be extracted from a variable by using the ``display_data()`` and 
``get_base_data()`` methods.  The former will extract the data in the 
assigned unit, while the latter will extract the data in the base unit.

Setting Display Unit
--------------
The display unit of a variable may be changed using the ``set_display_unit()`` 
method.  This requires a class of the ``units`` module as an argument.

Setting Data
------------
The data of a variable may be changed using the ``set_data()`` method.  This 
requires a single value or pandas series object as an argument, depending on 
the variable type.  

Operations
----------
Variables with the same display unit can be added and subtracted using the "+" 
and "-" operands.  The result is a third variable with the resulting data, 
same display unit, and name as "var1_var2".

Classes
=======

.. autoclass:: mpcpy.variables.Static
    :members: set_data, display_data, get_base_data, get_display_unit, get_base_unit, set_display_unit, get_display_unit_name
.. autoclass:: mpcpy.variables.Timeseries
    :members: set_data, display_data, get_base_data, get_display_unit, get_base_unit, set_display_unit, get_display_unit_name, cleaning_replace

"""

from abc import ABCMeta, abstractmethod
from tzwhere import tzwhere
import numpy as np

#%% Variable abstract class
class Variable(object):
    '''Base class for variables.

    '''

    __metaclass__ = ABCMeta;
    
    @abstractmethod
    def set_data(self,data):
        '''Set the data of the variable including any conversions to be performed.
        
        '''
        
        pass;
        
    def display_data(self):
        '''Return the data of the variable in display units.
        
        Returns
        -------
        data : data object
            Data object of the variable in display units.
            
        '''

        self._display_data();
    
        return self._data;
        
    def get_base_data(self):
        '''Return the data of the variable in base units.

        Returns
        -------
        data : data object
            Data object of the variable in base units.

        '''

        return self.data;

    def set_display_unit(self, display_unit):
        '''Set the display unit of the variable.
        
        Parameters
        ----------        
        display_unit : mpcpy.units.unit
            Display unit to set.   

        '''
        
        quantity_old = self.quantity_name;
        self.display_unit = display_unit(self);
        if quantity_old != self.quantity_name:
            raise(TypeError, 'Display unit to be set has a different quantity than the existing variable display unit.');
            
    def get_base_unit(self):
        '''Returns the base unit of the variable.

        Returns
        -------
        base_unit : mpcpy.units.unit
            Base unit of variable.

        '''

        return self.base_unit;


    def get_display_unit(self):
        '''Returns the display unit of the variable.

        Returns
        -------
        display_unit : mpcpy.units.unit
            Display unit of variable.

        '''

        return type(self.display_unit);

    def get_display_unit_name(self):
        '''Returns the display unit name of the variable.

        Returns
        -------
        display_unit_name : string
            Display unit name of variable.

        '''

        return self.display_unit.name;

    def __str__(self):
        '''Returns variable name, variability, unit quantity, and display unit name.

        Returns
        -------
        string : string
            String of variable information.

        '''

        string = 'Name: ' + self.name + '\n';
        string += 'Variability: ' + self.variability + '\n';
        string += 'Quantity: ' + self.quantity_name + '\n';       
        string += 'Display Unit: ' + self.display_unit.name + '\n';

        return string

    def __add__(self, variable):
        '''Returns resulting variable of addition of other variables to self variable.

        Parameters
        ----------
        variable : Static or Timeseries
            Other variable with which to perform requested operation.

        Returns
        -------
        variable_out : Static or Timeseries
            Resulting variable from addition of two other variables.

        '''

        variable_out = self._perform_operation(variable, 'add');

        return variable_out;

    def __sub__(self, variable):
        '''Returns resulting variable of subtraction of other variable from self variable.

        Parameters
        ----------
        variable : Static or Timeseries
            Other variable with which to perform requested operation.

        Returns
        -------
        variable_out : Static or Timeseries
            Resulting variable from subtraction of two other variables.

        '''
        
        variable_out = self._perform_operation(variable, 'sub');

        return variable_out;        

    def _perform_operation(self, variable, operation):
        '''Perform operation of addition or subtraction.
        
        Parameters
        ----------
        variable : Static or Timeseries
            Other variable with which to perform requested operation.
        operation : string
            Request addition ('add') or subtraction ('sub').
            
        Returns
        -------
        variable_out : Static or Timeseries
            Resulting variable from operation.

        '''

        if self.display_unit.name == variable.display_unit.name:        
            data1 = self.display_data();
            data2 = variable.display_data();
            if operation == 'add':
                data3 = data1 + data2;
            elif operation == 'sub':
                data3 = data1 - data2;              
            if self.variability == 'Timeseries' or variable.variability == 'Timeseries':  
                variable_out = Timeseries(self.name+variable.name, data3, self.get_display_unit());
            else:
                variable_out = Static(self.name+variable.name, data3, self.get_display_unit());  
                
            return variable_out

        else:
            raise(TypeError, 'Display units {} and {} are not the same.  Cannot perform operation.'.format(self.display_unit.name, variable.display_unit.name));

#%% Variable implementations
class Static(Variable):
    '''Variable class with data that is not a timeseries.
    
    Parameters
    ----------
    name : string
        Name of variable.
    data : float, int, bool, list, ``numpy`` array
        Data of variable
    display_unit : mpcpy.units.unit
        Unit of variable data being set.
        
    Attributes
    ----------
    name : string
        Name of variable.
    data : float, int, bool, list, ``numpy`` array
        Data of variable
    display_unit : mpcpy.units.unit
        Unit of variable data when returned with ``display_data()``.
    quantity_name : string
        Quantity type of the variable (e.g. Temperature, Power, etc.).        
    variability : string
        Static.

    '''

    def __init__(self, name, data, display_unit):
        '''Constructor of Static variable object.
        
        '''
        
        self.name = name;
        self.variability = 'Static';
        self.display_unit = display_unit(self);
        self.set_data(data);

    def set_data(self, data):
        '''Set data of Static variable.
        
        Parameters
        ----------
        data : float, int, bool, list, ``numpy`` array
            Data to be set for variable.

        Yields
        ----------
        data : float, int, bool, list, ``numpy`` array
            Data attribute.            

        '''

        if isinstance(data, float):
            self.data = self.display_unit._convert_to_base(float(data));
        elif isinstance(data, int): 
            self.data = self.display_unit._convert_to_base(float(data));
        elif isinstance(data, list):
            self.data = [self.display_unit._convert_to_base(float(x)) for x in data];
        elif isinstance(data, np.ndarray):
            self.data = np.array([self.display_unit._convert_to_base(float(x)) for x in data]);
        else:
            self.data = self.display_unit._convert_to_base(data);

    def _display_data(self):
        '''Return the data of the variable in display units.

        '''

        if isinstance(self.data, list):
            self._data = [self.display_unit._convert_from_base(x) for x in self.data];
        else:
            self._data = self.display_unit._convert_from_base(self.data);

class Timeseries(Variable):
    '''Variable class with data that is a timeseries.

    Parameters
    ----------
    name : string
        Name of variable.
    timeseries : ``pandas`` Series
        Timeseries data of variable.  Must have an index of timestamps.
    display_unit : mpcpy.units.unit
        Unit of variable data being set.
    tz_name : string
        Timezone name according to ``tzwhere``.
    geography : list, optional
        List specifying [latitude, longitude] in degrees.
    cleaning_type : dict, optional
        Dictionary specifying {'cleaning_type' : mpcpy.variables.Timeseries.cleaning_type, 'cleaning_args' : cleaning_args}.

    Attributes
    ----------
    name : string
        Name of variable.
    data : float, int, bool, list, ``numpy`` array
        Data of variable
    display_unit : mpcpy.units.unit
        Unit of variable data when returned with ``display_data()``.
    quantity_name : string
        Quantity type of the variable (e.g. Temperature, Power, etc.).        
    variability : string
        Timeseries.

    '''

    def __init__(self, name, timeseries, display_unit, tz_name = 'UTC', **kwargs):
        '''Constructor of Timeseries variable object.

        '''

        self.variability = 'Timeseries';
        self.display_unit = display_unit(self);
        self.set_data(timeseries, tz_name, **kwargs);
        self.name = name; 

    def set_data(self, timeseries, tz_name = 'UTC', **kwargs):
        '''Set data of Timeseries variable.

        Parameters
        ----------
        data : ``pandas`` Series
            Timeseries data of variable.  Must have an index of timestamps.
        tz_name : string
            Timezone name according to ``tzwhere``.
        geography : list, optional
            List specifying [latitude, longitude] in degrees.
        cleaning_type : dict, optional
            Dictionary specifying {'cleaning_type' : mpcpy.variables.Timeseries.cleaning_type, 'cleaning_args' : cleaning_args}.

        Yields
        ------
        data : ``pandas`` Series
            Data attribute.            

        '''

        self._timeseries = timeseries;       
        if 'cleaning_type' in kwargs and kwargs['cleaning_type'] is not None:       
            cleaning_type = kwargs['cleaning_type'];
            cleaning_args = kwargs['cleaning_args'];            
            self._timeseries = cleaning_type(self, cleaning_args)
        if 'geography' in kwargs:
            self._load_time_zone(kwargs['geography']);
            self._timeseries = self._local_to_utc(self._timeseries);
        else:
            self.tz_name = tz_name;
            self._timeseries = self._local_to_utc(self._timeseries);
            
        self.data = self.display_unit._convert_to_base(self._timeseries.apply(float));

    def _display_data(self):
        '''Return the data of the variable in display units.

        Yields
        ------
        _data
            Internal data variable.
        
        '''
        
        self._data = self.display_unit._convert_from_base(self.data);
        self._data = self._utc_to_local(self._data);

    def _local_to_utc(self, df_local):
        '''Convert a ``pandas`` Series in local time to utc time.

        Parameters
        ----------
        df_local
            Series in local time.
            
        Returns
        -------
        df_utc
            Series in utc time.

        '''

        try:
            df_local = df_local.tz_localize(self.tz_name);
            df_utc = df_local.tz_convert('UTC');
        except TypeError:
            df_utc = df_local.tz_convert('UTC');
            
        return df_utc;

    def _utc_to_local(self, df_utc):
        '''Convert a ``pandas`` Series in utc time to local time.

        Parameters
        ----------
        df_utc
            Series in utc time.

        Returns
        -------
        df_local
            Series in local time.

        '''

        df_local = df_utc.tz_convert(self.tz_name);

        return df_local;        

    def _load_time_zone(self, geography):
        '''Load the time zone name from geography.

        Parameters
        ----------
        geography : list
            List specifying [latitude, longitude] in degrees.

        Yields
        ------
        tz_name : string
            Timezone attribute named according to ``tzwhere``.

        '''

        try:
            self.tz_name = self.tz.tzNameAt(geography[0], geography[1]);
        except AttributeError:
            self.tz = tzwhere.tzwhere();
            self.tz_name = self.tz.tzNameAt(geography[0], geography[1]);        

    def cleaning_replace(self, (to_replace, replace_with)):
        '''Cleaning method to replace values within timeseries.

        Parameters
        ----------
        to_replace
            Value to replace.
        replace_with
            Replacement value.
            
        Returns
        -------
        timeseries
            Timeseries with data replaced according to to_replace and replace_with.        

        '''

        timeseries = self._timeseries.replace(to_replace,replace_with);

        return timeseries