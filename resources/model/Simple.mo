within ;
package Simple "A package containing simple examples"
  model RC "A simple RC network for example purposes"
    Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature
      prescribedTemperature
      annotation (Placement(transformation(extent={{-40,0},{-20,20}})));
    Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor temperatureSensor
      annotation (Placement(transformation(extent={{70,0},{90,20}})));
    Modelica.Blocks.Sources.Sine Tamb(
      amplitude=10,
      offset=278.15,
      freqHz=1/(24*3600))
      annotation (Placement(transformation(extent={{-80,0},{-60,20}})));
    Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow prescribedHeatFlow
      annotation (Placement(transformation(extent={{-18,-40},{2,-20}})));
    Modelica.Blocks.Interfaces.RealOutput T_db
      "Absolute temperature as output signal"
      annotation (Placement(transformation(extent={{100,0},{120,20}})));
    Modelica.Blocks.Interfaces.RealInput q_flow
      annotation (Placement(transformation(extent={{-140,-50},{-100,-10}})));
    Modelica.Thermal.HeatTransfer.Components.HeatCapacitor heatCapacitor(C=1e5)
      annotation (Placement(transformation(extent={{20,10},{40,30}})));
    Modelica.Thermal.HeatTransfer.Components.ThermalResistor thermalResistor(R=
          0.01) annotation (Placement(transformation(extent={{-10,0},{10,20}})));
  equation
    connect(temperatureSensor.T,T_db)
      annotation (Line(points={{90,10},{110,10}},       color={0,0,127}));
    connect(Tamb.y,prescribedTemperature. T)
      annotation (Line(points={{-59,10},{-42,10}},
                                                 color={0,0,127}));
    connect(prescribedHeatFlow.Q_flow,q_flow)
      annotation (Line(points={{-18,-30},{-120,-30}}, color={0,0,127}));
    connect(heatCapacitor.port, temperatureSensor.port)
      annotation (Line(points={{30,10},{50,10},{70,10}}, color={191,0,0}));
    connect(heatCapacitor.port, prescribedHeatFlow.port)
      annotation (Line(points={{30,10},{30,-30},{2,-30}}, color={191,0,0}));
    connect(prescribedTemperature.port, thermalResistor.port_a)
      annotation (Line(points={{-20,10},{-16,10},{-10,10}}, color={191,0,0}));
    connect(thermalResistor.port_b, heatCapacitor.port)
      annotation (Line(points={{10,10},{20,10},{30,10}}, color={191,0,0}));
    annotation ();
  end RC;
  annotation (uses(Modelica(version="3.2.2")));
end Simple;
