within ;
model VariableRadiation

  parameter Real Area(unit="m2") "Net radiation area between two surfaces";
  Real Q_flow "Heat flow between the ports";
  Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_a port_a
    annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
  Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_b port_b
    annotation (Placement(transformation(extent={{90,-10},{110,10}})));
  Modelica.Blocks.Interfaces.RealInput Eps annotation (Placement(transformation(
        extent={{-20,-20},{20,20}},
        rotation=270,
        origin={0,100})));
equation
  // Energy balance equations for the thermal ports
  port_a.Q_flow + port_b.Q_flow = 0;  // Conservation of energy
  Q_flow = port_a.Q_flow;            // Define Q_flow as the heat flow through port_a

  // Thermal radiation equation
  Q_flow = Area * Eps * Modelica.Constants.sigma * (port_a.T^4 - port_b.T^4);

      // HeatPort_a visualization
      // HeatPort_b visualization
      // Block body
      // RealInput port visualization as a triangle
annotation (
    Icon(coordinateSystem(preserveAspectRatio=true, extent={{-100,-100},{
            100,100}}), graphics={
        Rectangle(
          extent={{50,80},{90,-80}},
          fillColor={192,192,192},
          fillPattern=FillPattern.Backward),
        Rectangle(
          extent={{-90,80},{-50,-80}},
          fillColor={192,192,192},
          fillPattern=FillPattern.Backward),
        Line(points={{-36,10},{36,10}}, color={191,0,0}),
        Line(points={{-36,10},{-26,16}}, color={191,0,0}),
        Line(points={{-36,10},{-26,4}}, color={191,0,0}),
        Line(points={{-36,-10},{36,-10}}, color={191,0,0}),
        Line(points={{26,-16},{36,-10}}, color={191,0,0}),
        Line(points={{26,-4},{36,-10}}, color={191,0,0}),
        Line(points={{-36,-30},{36,-30}}, color={191,0,0}),
        Line(points={{-36,-30},{-26,-24}}, color={191,0,0}),
        Line(points={{-36,-30},{-26,-36}}, color={191,0,0}),
        Line(points={{-36,30},{36,30}}, color={191,0,0}),
        Line(points={{26,24},{36,30}}, color={191,0,0}),
        Line(points={{26,36},{36,30}}, color={191,0,0}),
        Text(
          extent={{-150,125},{150,85}},
          textString="%name",
          textColor={0,0,255}),
        Text(
          extent={{-150,-90},{150,-120}},
          textString="Area=%Area"),
        Rectangle(
          extent={{-50,80},{-44,-80}},
          lineColor={191,0,0},
          fillColor={191,0,0},
          fillPattern=FillPattern.Solid),
        Rectangle(
          extent={{45,80},{50,-80}},
          lineColor={191,0,0},
          fillColor={191,0,0},
          fillPattern=FillPattern.Solid)}),
  Documentation(info="<html>
<p>
This is a model describing the thermal radiation, i.e., electromagnetic
radiation emitted between two bodies as a result of their temperatures.
The following constitutive equation is used:
</p>
<blockquote><pre>
Q_flow = Area * Eps * sigma * (port_a.T^4 - port_b.T^4);
</pre></blockquote>
<p>
where Area is the area subject to radiation, Eps is the variable emissivity of the surface, and sigma is the Stefan-Boltzmann constant (= Modelica.Constants.sigma). 
</html>"),
  uses(Modelica(version="4.0.0")));
end VariableRadiation;
