model Assignment1
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Rad1(C = 30, T(start = 298.15, fixed = true, displayUnit = "K")) annotation (
    Placement(transformation(origin = {-150, -90}, extent = {{-10, -10}, {10, 10}}, rotation = 90)));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor Rad2(C = 30, T(start = 298.15, fixed = true, displayUnit = "K")) annotation (
    Placement(transformation(origin = {152, -90}, extent = {{-10, -10}, {10, 10}}, rotation = 270)));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor G14(G = 10) annotation (
    Placement(transformation(origin = {20, 40}, extent = {{-140, -140}, {-120, -120}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor G12(G = 10) annotation (
    Placement(transformation(origin = {4, 0}, extent = {{-124, 140}, {-104, 160}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor G13(G = 10) annotation (
    Placement(transformation(origin = {-54, 0}, extent = {{154, 140}, {174, 160}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalConductor G15(G = 10) annotation (
    Placement(transformation(origin = {-20, 40}, extent = {{120, -140}, {140, -120}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor SA1(C = 1187.5, T(start = 298.15, fixed = true, displayUnit = "K")) annotation (
    Placement(transformation(origin = {-150, 150}, extent = {{-10, -10}, {10, 10}}, rotation = 90)));
  Modelica.Thermal.HeatTransfer.Components.BodyRadiation DeepSpaceRadiation2(Gr = 0.37425) annotation (
    Placement(transformation(origin = {-110, 190}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor SA2(C = 1187.5, T(start = 298.15, fixed = true, displayUnit = "K")) annotation (
    Placement(transformation(origin = {150, 150}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor MainBody(C = 150000, T(start = 298.15, fixed = true, displayUnit = "K")) annotation (
    Placement(transformation(origin = {0, 60}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedHeatFlow SolarHeatFlow2(Q_flow = 500.175) annotation (
    Placement(transformation(origin = {-110, 110}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature DeepSpaceTemperature2(T(displayUnit = "K") = 3) annotation (
    Placement(transformation(origin = {-70, 190}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
  Modelica.Thermal.HeatTransfer.Components.BodyRadiation DeepSpaceRadiation3(Gr = 0.37425) annotation (
    Placement(transformation(origin = {110, 190}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature DeepSpaceTemperature3(T(displayUnit = "K") = 3) annotation (
    Placement(transformation(origin = {70, 190}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedHeatFlow SolarHeatFlow3(Q_flow = 500.175) annotation (
    Placement(transformation(origin = {110, 110}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedHeatFlow SolarHeatFlow1(Q_flow = 202.5) annotation (
    Placement(transformation(origin = {30, -10}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
  Modelica.Thermal.HeatTransfer.Components.BodyRadiation DeepSpaceRadiation1(Gr = 1.458) annotation (
    Placement(transformation(origin = {30, 30}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature DeepSpaceTemperature1(T(displayUnit = "K") = 3) annotation (
    Placement(transformation(origin = {70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
  VariableRadiation VariableRadiation2(Area = 0.5) annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 180, origin = {110, -130})));
  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature DeepSpaceTemperature5(T(displayUnit = "K") = 3) annotation (
    Placement(transformation(extent = {{60, -140}, {80, -120}})));
  VariableRadiation VariableRadiation1(Area = 0.5) annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 0, origin = {-110, -130})));
  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature DeepSpaceTemperature4(T(displayUnit = "K") = 3) annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 180, origin = {-70, -130})));
  Modelica.Thermal.HeatTransfer.Sensors.TemperatureSensor T1 annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 180, origin = {-70, 30})));
  Modelica.Electrical.Analog.Basic.Resistor resistor(R = 0.1) annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 0, origin={-254,-90})));
  Modelica.Electrical.Analog.Basic.Inductor inductor(L = 0.001) annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 0, origin={-294,-90})));
  Modelica.Electrical.Analog.Sources.SignalVoltage ProportionalVoltage annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 0, origin={-254,-50})));
  Modelica.Electrical.Analog.Basic.Ground ground annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 180, origin={-314,-40})));
  Modelica.Blocks.Math.Gain Kp(k = 2e-4) annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 180, origin={-150,10})));
  Modelica.Blocks.Math.Add add(k1 = -1, k2 = +1) annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 180, origin = {-112, 10})));
  Modelica.Blocks.Sources.Constant ReferenceTemperature(k = 294.15) annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 180, origin = {-70, -10})));
  Modelica.Mechanics.Rotational.Sensors.AngleSensor Theta annotation (
    Placement(transformation(extent = {{-260, -200}, {-240, -180}})));
  Modelica.Mechanics.MultiBody.Joints.Revolute Hinge(useAxisFlange = true, phi(start = -0.4*Modelica.Constants.pi, fixed = true)) annotation (
    Placement(transformation(extent = {{-300, -240}, {-280, -220}})));
  Modelica.Mechanics.MultiBody.Parts.Body Radiator(animation = false, m = 0.2, I_11 = 0.00833, I_22 = 0.00833, I_33 = 0.01667) annotation (
    Placement(transformation(extent = {{-260, -240}, {-240, -220}})));
  Modelica.Electrical.Analog.Basic.RotationalEMF emf(k = 0.3) annotation (
    Placement(transformation(extent = {{-10, -11}, {10, 11}}, rotation = 0, origin={-334,-71})));
  Modelica.Blocks.Sources.Constant EpsMin(k = 0.01) annotation (
    Placement(transformation(extent = {{-140, -260}, {-120, -240}})));
  Modelica.Blocks.Sources.Constant EpsMax(k = 0.98) annotation (
    Placement(transformation(extent = {{-140, -300}, {-120, -280}})));
  Modelica.Blocks.Sources.Constant Pi(k = Modelica.Constants.pi) annotation (
    Placement(transformation(extent = {{-180, -220}, {-160, -200}})));
  Modelica.Blocks.Math.Add add2 annotation (
    Placement(transformation(extent = {{-100, -210}, {-80, -190}})));
  Modelica.Blocks.Math.Gain gain(k = 0.4) annotation (
    Placement(transformation(extent = {{-140, -220}, {-120, -200}})));
  Modelica.Blocks.Math.Add add1(k1 = -1) annotation (
    Placement(transformation(extent = {{-100, -280}, {-80, -260}})));
  Modelica.Blocks.Math.Division division annotation (
    Placement(transformation(extent = {{-60, -260}, {-40, -240}})));
  Modelica.Blocks.Math.Product product1 annotation (
    Placement(transformation(extent = {{-20, -220}, {0, -200}})));
  Modelica.Blocks.Math.Add add3 annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 90, origin = {30, -190})));
  Modelica.Blocks.Logical.Switch switch1 annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 270, origin={-310,70})));
  Modelica.Blocks.Sources.Constant const(k = 0) annotation (
    Placement(transformation(extent={{-362,90},{-342,110}})));
  Modelica.Blocks.Logical.And and1 annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 180, origin = {-270, 70})));
  Modelica.Blocks.Logical.And and2 annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 180, origin = {-270, 150})));
  Modelica.Blocks.Logical.GreaterThreshold T1vsTref(threshold = 294.15) annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 180, origin = {-230, 172})));
  Modelica.Blocks.Logical.GreaterThreshold ThetaMax(threshold = 0) annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 180, origin = {-230, 130})));
  Modelica.Blocks.Logical.Or or1 annotation (
    Placement(transformation(origin={-310,130},   extent = {{-10, -10}, {10, 10}}, rotation = 270)));
  Modelica.Blocks.Logical.LessThreshold ThetaMin(threshold = -0.4*Modelica.Constants.pi) annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 180, origin = {-230, 50})));
  Modelica.Blocks.Logical.LessThreshold T1vsTref1(threshold = 294.15) annotation (
    Placement(transformation(extent = {{-10, -10}, {10, 10}}, rotation = 180, origin = {-230, 90})));
  inner Modelica.Mechanics.MultiBody.World world(animateGravity = false, gravityType = Modelica.Mechanics.MultiBody.Types.GravityTypes.NoGravity) annotation (
    Placement(transformation(extent = {{-340, -240}, {-320, -220}})));
equation
  connect(G14.port_a, Rad1.port) annotation (
    Line(points = {{-120, -90}, {-140, -90}}, color = {191, 0, 0}));
  connect(G15.port_b, Rad2.port) annotation (
    Line(points = {{120, -90}, {142, -90}}, color = {191, 0, 0}));
  connect(SA1.port, G12.port_a) annotation (
    Line(points = {{-140, 150}, {-120, 150}}, color = {191, 0, 0}));
  connect(SA1.port, DeepSpaceRadiation2.port_a) annotation (
    Line(points = {{-140, 150}, {-140, 190}, {-120, 190}}, color = {191, 0, 0}));
  connect(G13.port_b, SA2.port) annotation (
    Line(points = {{120, 150}, {140, 150}}, color = {191, 0, 0}));
  connect(G12.port_b, MainBody.port) annotation (
    Line(points = {{-100, 150}, {-16, 150}, {-16, 50}, {0, 50}}, color = {191, 0, 0}));
  connect(G13.port_a, MainBody.port) annotation (
    Line(points = {{100, 150}, {20, 150}, {20, 50}, {0, 50}}, color = {191, 0, 0}));
  connect(SA1.port, SolarHeatFlow2.port) annotation (
    Line(points = {{-140, 150}, {-140, 110}, {-120, 110}, {-120, 110}}, color = {191, 0, 0}));
  connect(DeepSpaceTemperature2.port, DeepSpaceRadiation2.port_b) annotation (
    Line(points = {{-80, 190}, {-100, 190}}, color = {191, 0, 0}));
  connect(DeepSpaceRadiation3.port_a, SA2.port) annotation (
    Line(points = {{120, 190}, {140, 190}, {140, 150}}, color = {191, 0, 0}));
  connect(DeepSpaceRadiation3.port_b, DeepSpaceTemperature3.port) annotation (
    Line(points = {{100, 190}, {80, 190}}, color = {191, 0, 0}));
  connect(SolarHeatFlow3.port, SA2.port) annotation (
    Line(points = {{120, 110}, {140, 110}, {140, 150}}, color = {191, 0, 0}));
  connect(SolarHeatFlow1.port, MainBody.port) annotation (
    Line(points = {{20, -10}, {0, -10}, {0, 50}}, color = {191, 0, 0}));
  connect(DeepSpaceRadiation1.port_a, MainBody.port) annotation (
    Line(points = {{20, 30}, {0, 30}, {0, 50}}, color = {191, 0, 0}));
  connect(DeepSpaceTemperature1.port, DeepSpaceRadiation1.port_b) annotation (
    Line(points = {{60, 30}, {40, 30}}, color = {191, 0, 0}));
  connect(G15.port_a, MainBody.port) annotation (
    Line(points = {{100, -90}, {0, -90}, {0, 50}}, color = {191, 0, 0}));
  connect(G14.port_b, MainBody.port) annotation (
    Line(points = {{-100, -90}, {0, -90}, {0, 50}}, color = {191, 0, 0}));
  connect(VariableRadiation2.port_a, Rad2.port) annotation (
    Line(points = {{120, -130}, {142, -130}, {142, -90}}, color = {191, 0, 0}));
  connect(DeepSpaceTemperature5.port, VariableRadiation2.port_b) annotation (
    Line(points = {{80, -130}, {100, -130}}, color = {191, 0, 0}));
  connect(Rad1.port, VariableRadiation1.port_a) annotation (
    Line(points = {{-140, -90}, {-140, -130}, {-120, -130}}, color = {191, 0, 0}));
  connect(DeepSpaceTemperature4.port, VariableRadiation1.port_b) annotation (
    Line(points = {{-80, -130}, {-100, -130}}, color = {191, 0, 0}));
  connect(T1.port, MainBody.port) annotation (
    Line(points = {{-60, 30}, {0, 30}, {0, 50}}, color = {191, 0, 0}));
  connect(ReferenceTemperature.y, add.u1) annotation (
    Line(points = {{-81, -10}, {-88, -10}, {-88, 4}, {-100, 4}}, color = {0, 0, 127}));
  connect(T1.T, add.u2) annotation (
    Line(points = {{-81, 30}, {-88, 30}, {-88, 16}, {-100, 16}}, color = {0, 0, 127}));
  connect(add.y, Kp.u) annotation (
    Line(points={{-123,10},{-138,10}},      color = {0, 0, 127}));
  connect(Radiator.frame_a, Hinge.frame_b) annotation (
    Line(points = {{-260, -230}, {-280, -230}}, color = {95, 95, 95}, thickness = 0.5));
  connect(Theta.flange, Hinge.axis) annotation (
    Line(points = {{-260, -190}, {-290, -190}, {-290, -220}}, color = {0, 0, 0}));
  connect(Pi.y, gain.u) annotation (
    Line(points = {{-159, -210}, {-142, -210}}, color = {0, 0, 127}));
  connect(Theta.phi, add2.u1) annotation (
    Line(points = {{-239, -190}, {-110, -190}, {-110, -194}, {-102, -194}}, color = {0, 0, 127}));
  connect(gain.y, add2.u2) annotation (
    Line(points = {{-119, -210}, {-112, -210}, {-112, -206}, {-102, -206}}, color = {0, 0, 127}));
  connect(EpsMax.y, add1.u2) annotation (
    Line(points = {{-119, -290}, {-110, -290}, {-110, -276}, {-102, -276}}, color = {0, 0, 127}));
  connect(EpsMin.y, add1.u1) annotation (
    Line(points = {{-119, -250}, {-110, -250}, {-110, -264}, {-102, -264}}, color = {0, 0, 127}));
  connect(gain.y, division.u2) annotation (
    Line(points = {{-119, -210}, {-119, -220}, {-72, -220}, {-72, -256}, {-62, -256}}, color = {0, 0, 127}));
  connect(add1.y, division.u1) annotation (
    Line(points = {{-79, -270}, {-78, -270}, {-78, -244}, {-62, -244}}, color = {0, 0, 127}));
  connect(add2.y, product1.u1) annotation (
    Line(points = {{-79, -200}, {-40, -200}, {-40, -204}, {-22, -204}}, color = {0, 0, 127}));
  connect(division.y, product1.u2) annotation (
    Line(points = {{-39, -250}, {-39, -216}, {-22, -216}}, color = {0, 0, 127}));
  connect(product1.y, add3.u1) annotation (
    Line(points = {{1, -210}, {24, -210}, {24, -202}}, color = {0, 0, 127}));
  connect(EpsMin.y, add3.u2) annotation (
    Line(points = {{-119, -250}, {-110, -250}, {-110, -230}, {36, -230}, {36, -202}}, color = {0, 0, 127}));
  connect(add3.y, VariableRadiation2.Eps) annotation (
    Line(points = {{30, -179}, {30, -160}, {110, -160}, {110, -140}}, color = {0, 0, 127}));
  connect(add3.y, VariableRadiation1.Eps) annotation (
    Line(points = {{30, -179}, {30, -160}, {-20, -160}, {-20, -106}, {-110, -106}, {-110, -120}}, color = {0, 0, 127}));
  connect(resistor.n, ProportionalVoltage.n) annotation (
    Line(points={{-244,-90},{-238,-90},{-238,-50},{-244,-50}},              color = {0, 0, 255}));
  connect(resistor.p, inductor.n) annotation (
    Line(points={{-264,-90},{-284,-90}},        color = {0, 0, 255}));
  connect(inductor.p, emf.n) annotation (
    Line(points={{-304,-90},{-334,-90},{-334,-82}},           color = {0, 0, 255}));
  connect(emf.p, ground.p) annotation (
    Line(points={{-334,-60},{-334,-50},{-314,-50}},           color = {0, 0, 255}));
  connect(ground.p, ProportionalVoltage.p) annotation (
    Line(points={{-314,-50},{-264,-50}},        color = {0, 0, 255}));
  connect(emf.flange, Hinge.axis) annotation (
    Line(points={{-324,-71},{-312,-71},{-312,-212},{-290,-212},{-290,-220}},color = {0, 0, 0}));
  connect(switch1.y, ProportionalVoltage.v) annotation (
    Line(points={{-310,59},{-312,59},{-312,0},{-254,0},{-254,-38}},
                                              color = {0, 0, 127}));
  connect(T1.T, T1vsTref.u) annotation (
    Line(points = {{-81, 30}, {-200, 30}, {-200, 172}, {-218, 172}}, color = {0, 0, 127}));
  connect(T1vsTref.y, and2.u2) annotation (
    Line(points = {{-241, 172}, {-248, 172}, {-248, 158}, {-258, 158}}, color = {255, 0, 255}));
  connect(Theta.phi, ThetaMax.u) annotation (
    Line(points = {{-239, -190}, {-210, -190}, {-210, 130}, {-218, 130}}, color = {0, 0, 127}));
  connect(ThetaMax.y, and2.u1) annotation (
    Line(points = {{-241, 130}, {-248, 130}, {-248, 150}, {-258, 150}}, color = {255, 0, 255}));
  connect(or1.y, switch1.u2) annotation (
    Line(points={{-310,119},{-310,82}},      color = {255, 0, 255}));
  connect(T1vsTref1.y, and1.u2) annotation (
    Line(points = {{-241, 90}, {-250, 90}, {-250, 78}, {-258, 78}}, color = {255, 0, 255}));
  connect(ThetaMin.y, and1.u1) annotation (
    Line(points = {{-241, 50}, {-250, 50}, {-250, 70}, {-258, 70}}, color = {255, 0, 255}));
  connect(T1.T, T1vsTref1.u) annotation (
    Line(points = {{-81, 30}, {-200, 30}, {-200, 90}, {-218, 90}}, color = {0, 0, 127}));
  connect(Theta.phi, ThetaMin.u) annotation (
    Line(points = {{-239, -190}, {-210, -190}, {-210, 50}, {-218, 50}}, color = {0, 0, 127}));
  connect(and1.y, or1.u1) annotation (
    Line(points={{-281,70},{-292,70},{-292,148},{-310,148},{-310,142}},
                                                        color = {255, 0, 255}));
  connect(and2.y, or1.u2) annotation (
    Line(points={{-281,150},{-318,150},{-318,142}},       color = {255, 0, 255}));
  connect(Hinge.frame_a, world.frame_b) annotation (
    Line(points = {{-300, -230}, {-320, -230}}, color = {95, 95, 95}, thickness = 0.5));
  connect(const.y, switch1.u1) annotation (
    Line(points={{-341,100},{-302,100},{-302,82}},         color = {0, 0, 127}));
  connect(Kp.y, switch1.u3) annotation (
    Line(points={{-161,10},{-286,10},{-286,94},{-318,94},{-318,82}},               color = {0, 0, 127}));
  annotation (
    Diagram(coordinateSystem(extent = {{-480, -360}, {250, 250}}), graphics={  Bitmap(extent = {{-220, 220}, {-214, 220}}), Rectangle(origin = {-110, 151}, lineColor = {28, 108, 200}, fillColor = {226, 226, 226}, fillPattern = FillPattern.Solid, lineThickness = 1, extent = {{-60, 61}, {60, -61}}), Text(origin={-109,219},    textColor = {28, 108, 200}, extent={{59,-5},
              {-59,5}},                                                                                                                                                                                                        textString = "Solar Array 1 (Thermal Node 2)"), Rectangle(origin = {110, 151}, lineColor = {28, 108, 200}, fillColor = {226, 226, 226}, fillPattern = FillPattern.Solid, lineThickness = 1, extent = {{-60, 61}, {60, -61}}), Text(origin={111,218},    textColor = {28, 108, 200}, extent={{59,-6},
              {-59,6}},                                                                                                                                                                                                        textString = "Solar Array 2 (Thermal Node 3)"), Rectangle(origin = {31, 25}, lineColor = {28, 108, 200}, fillColor = {226, 226, 226}, fillPattern = FillPattern.Solid, lineThickness = 1, extent = {{-59, 55}, {59, -55}}), Text(origin={62,-37},    textColor = {28, 108, 200}, extent={{58,-5},
              {-58,5}},                                                                                                                                                                                                        textString = "Main Body (Thermal Node 1)"), Rectangle(origin = {110, -110}, lineColor = {28, 108, 200}, fillColor = {226, 226, 226}, fillPattern = FillPattern.Solid, lineThickness = 1, extent = {{-60, 40}, {60, -40}}), Text(origin={117,-64},    textColor = {28, 108, 200}, extent={{55,-6},
              {-55,6}},                                                                                                                                                                                                        textString = "Radiator 2 (Thermal Node 5)"), Rectangle(origin = {-110, -110}, lineColor = {28, 108, 200}, fillColor = {226, 226, 226}, fillPattern = FillPattern.Solid, lineThickness = 1, extent = {{-60, 40}, {60, -40}}), Text(origin={-104,-64},   textColor = {28, 108, 200}, extent={{56,-8},
              {-56,8}},                                                                                                                                                                                                        textString = "Radiator 1 (Thermal Node 4)"), Rectangle(origin = {-70, -240}, lineColor = {255, 128, 0}, fillColor = {226, 226, 226}, fillPattern = FillPattern.Solid, lineThickness = 1, extent = {{-120, 70}, {120, -70}}), Text(origin = {-142, -319}, textColor = {244, 125, 35}, extent = {{48, -7}, {-48, 7}}, textString = "Emissivity Calculation"), Rectangle(lineColor = {255, 170, 213}, fillColor = {226, 226, 226}, fillPattern = FillPattern.Solid, lineThickness = 1, extent={{-372,
              190},{-190,28}}),                                                                                                                                                                                                        Text(origin = {-334, 198}, textColor = {217, 67, 180}, extent = {{52, -6}, {-52, 6}}, textString = "Control Logic"), Rectangle(lineColor = {0, 127, 0}, fillColor = {226, 226, 226}, fillPattern = FillPattern.Solid, lineThickness = 1, extent={{-354,
              -30},{-224,-110}}),                                                                                                                                                                                                        Text(origin={-319,-23},    textColor = {0, 140, 72}, extent = {{37, -5}, {-37, 5}}, textString = "Electrical Circuit"),
                                                                                                                                                                                                        Rectangle(lineColor={102,44,
              145},                                                                                                                                                                                                        fillColor = {226, 226, 226}, fillPattern = FillPattern.Solid, lineThickness = 1, extent={{-350,
              -176},{-230,-250}}),                                                                                                                                                                                                        Text(origin={-315,
              -257},                                                                                                                                                                                                        textColor={102,44,
              145},                                                                                                                                                                                                        extent = {{37, -5}, {-37, 5}},
          textString="Mechanical Hinge")}),
    Icon(coordinateSystem(extent = {{-480, -360}, {250, 250}})),
    uses(Modelica(version = "4.0.0")),
    version = "");
end Assignment1;
