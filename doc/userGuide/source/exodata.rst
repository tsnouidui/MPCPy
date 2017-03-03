=======
ExoData
=======

.. automodule:: mpcpy.exodata

Weather
=======

.. autoclass:: mpcpy.exodata.Weather

Internal
========

Internal data represents zone heat gains that may come from people, lights, or equipment.  Internal data objects have special methods for sourcing these heat gains from a predicted occupancy model.

Structure
---------

Exogenous internal data has the following organization:

::

    internal.data = {"Zone Name" : {
                        "Internal Variable Name" : mpcpy.Variables.Timeseries}}

The internal variable names should be chosen from the following list:

- intCon - convective internal load
- intRad - radiative internal load
- intLat - latent internal load

The internal variable names input in the model should follow the convention ``internalVariableName_zoneName``.  For example, the convective load input for the zone "west" should have the name ``intCon_west``.

Classes
-------

Internal data may be collected using the following classes:

    InternalFromCSV
    
        Collects internal data from a CSV file.  This class requires a variable map to match CSV column headers with internal variable names.  The variable map is a python dictionary of the form: 

::

    variable_map = {"Column Header Name" : ("Zone Name", 
                                            "Internal Variable Name", 
                                            mpcpy.Units.unit)}
\
 
    InternalFromOccupancyModel
    
        Generates internal load data from an occupancy prediction model.  This class requires a zone list in the form ["Zone Name 1", "Zone Name 2", "Zone Name 3"], a list of numeric values representing the loads per person in the form [Convective, Radiative, Latent] for each zone and collected in a list, the units of the indicated loads from ``mpcpy.Units.unit``, and a list of occupancy model objects with predicted occupancy, one for each zone.


Control
=======

Control data represents control inputs to a system or model.  The variables listed in a Control data object are special in that they are considered optimization variables during model optimization.

Structure
---------

Exogenous control data has the following organization:

::

    control.data = {"Control Variable Name" : mpcpy.Variables.Timeseries}

The control variable names should match the control input variables of the model.

Classes
-------

Control data may be collected using the following classes:

    ControlFromCSV
    
        Collects control data from a CSV file.  This class requires a variable map to match CSV column headers with control variable names.  The variable map is a python dictionary of the form: 

::

    variable_map = {"Column Header Name" : ("Control Variable Name", 
                                            mpcpy.Units.unit)}


Other Inputs
============

Other Input data represents miscellaneous inputs to a model.  The variables listed in an Other Inputs data object are not acted upon in any special way.

Structure
---------

Other input data has the following organization:

::

    other_input.data = {"Other Input Variable Name" : mpcpy.Variables.Timeseries}

The other input variable names should match those of the model.

Classes
-------

Other input data may be collected using the following classes:

    OtherInputFromCSV
    
        Collect other input data from a CSV file.  This class requires a variable map to match CSV column headers with other input variable names.  The variable map is a python dictionary of the form: 

::

    variable_map = {"Column Header Name" : ("Other Input Variable Name", 
                                            mpcpy.Units.unit)}


Price
=====

Price data represents price signals from utility or district energy systems for things such as energy consumption, demand, or other services.  Price data object variables are special because they are used for optimization objective functions involving price signals.

Structure
---------

Exogenous price data has the following organization:

::

    price.data = {"Price Variable Name" : mpcpy.Variables.Timeseries}

The price variable names should be chosen from the following list:

- pi_e - electrical energy price

Classes
-------

Price data may be collected using the following classes:

    PriceFromCSV
    
        Collects price data from a CSV file.  This class requires a variable map to match CSV column headers with price variable names.  The variable map is a python dictionary of the form: 

::

    variable_map = {"Column Header Name" : ("Price Variable Name", 
                                            mpcpy.Units.unit)}


Constraints
===========

Constraint data represents limits to which the control and state variables of an optimization solution must abide.  Constraint data object variables are included in the optimization problem formulation.

Structure
---------

Exogenous constraint data has the following organization:

::

    constraint.data = {"State or Control Variable Name" : {
                            "Constraint Variable Name" : mpcpy.Variables.Timeseries/Static}}

The state or control variable name must match those that are in the model.  The constraint variable names should be chosen from the following list:

- LTE - less than or equal to (Timeseries)
- GTE - greater than or equal to (Timeseries)
- E - equal to (Timeseries)
- Initial - initial value (Static)
- Final - final value (Static)
- Cyclic - initial value equals final value (Static - Boolean)

Classes
-------

Constraint data may be collected using the following classes:

    ConstraintFromCSV
    
        Collects timeseries constraint data from a CSV file.  Static constraint data must be added by editing the data dictionary directly.  This class requires a variable map to match CSV column headers with constraint variable names.  The variable map is a python dictionary of the form: 

::

    variable_map = {"Column Header Name" : ("State or Control Variable Name", 
                                            "Constraint Variable Name", 
                                            mpcpy.Units.unit)}
\

    ConstraintFromOccupancyModel
        
        Generates LTE, GTE, and E constraint data from an occupancy prediction model by implementing occupied and unoccupied values.  This class requires a state or control variable list in the form ["Variable Name 1", "Variable Name 2", "Variable Name 3"], a list of numeric values representing the occupied and unoccupied constraint values in the form [Occupied, Unoccupied] for each variable collected in a list, a list of constraint variable names, one for each variable, and a list of the units of the indicated numeric values from ``mpcpy.Units.unit``.


Parameters
==========

Parameter data represents inputs or coefficients of models that do not change with time during a simulation, which may need to be learned using system measurement data. Parameter data object variables are set when simulating models, and are estimated using model learning techniques if flagged to do so.

Structure
---------

Exogenous parameter data has the following organization:

::

    parameter.data = {"Parameter Name" : {
                        "Parameter Variable Name" : mpcpy.Variables.Static}}

The parameter name must match that which is in the model.  The parameter variable names should be chosen from the following list:

- Free - boolean flag for inclusion in model learning algorithms
- Value - value of the parameter, which is also used as an initial guess for model learning algorithms
- Minimum - minimum value of the parameter for model learning algorithms
- Maximum - maximum value of the parameter for model learning algorithms
- Covariance - covariance of the parameter for model learning algorithms

Classes
-------

Parameter data may be collected using the following classes:

    ParameterFromCSV
    
        Collects parameter data from a CSV file.  The CSV file rows must be named as the parameter names and the columns must be named as the parameter variable names.