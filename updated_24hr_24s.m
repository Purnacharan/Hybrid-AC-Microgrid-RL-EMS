%% 24-hour microgrid profiles compressed into 24 seconds
% 1 second = 1 hour
% 0.25 second = 15 minutes

clear; clc; close all;

% Time vector
t = (0:0.25:24)';   % 0–24 seconds representing 24 hours

% ---------------- PV IRRADIANCE PROFILE ----------------
irr = zeros(size(t));

for k = 1:length(t)
    th = t(k);   % represents hour

    if th < 6
        irr(k) = 0;
    elseif th < 8
        irr(k) = 200 + (th-6)*(600-200)/2;
    elseif th < 12
        irr(k) = 600 + (th-8)*(1000-600)/4;
    elseif th < 14
        irr(k) = 1000 - (th-12)*(1000-850)/2;
    elseif th < 17
        irr(k) = 850 - (th-14)*(850-300)/3;
    elseif th < 18.5
        irr(k) = 300 - (th-17)*(300-0)/1.5;
    else
        irr(k) = 0;
    end
end

% Cloud variation
cloud_factor = 0.92 + 0.08*sin(2*pi*t/3.5);
irr = irr .* cloud_factor;

irr(irr < 0) = 0;
irr(irr > 1000) = 1000;

% ---------------- TEMPERATURE PROFILE ----------------
temp = zeros(size(t));

for k = 1:length(t)
    th = t(k);

    if th < 6
        temp(k) = 22;
    elseif th < 10
        temp(k) = 22 + (th-6)*(30-22)/4;
    elseif th < 14
        temp(k) = 30 + (th-10)*(38-30)/4;
    elseif th < 17
        temp(k) = 38 - (th-14)*(38-32)/3;
    elseif th < 21
        temp(k) = 32 - (th-17)*(32-25)/4;
    else
        temp(k) = 25;
    end
end

temp = temp + 0.8*sin(2*pi*t/5);
% ---------------- CREATE TIMESERIES ----------------
irr_ts  = timeseries(irr, t);
temp_ts = timeseries(temp, t);
% ===============================
% OBSERVATION SPACE
% ===============================

obsInfo = rlNumericSpec([5 1], ...
    LowerLimit = -inf, ...
    UpperLimit = inf);

obsInfo.Name = "microgrid_states";

% Observations:
% 1 Ppv
% 2 Pwind
% 3 Pload
% 4 SOC
% 5 Vdc

% ===============================
% ACTION SPACE (DISCRETE)
% ===============================

actInfo = rlFiniteSetSpec([1 2 3 4]);

actInfo.Name = "EMS_actions";

% Actions meaning:
% 1 = Charge battery
% 2 = Idle
% 3 = Discharge battery
% 4 = Turn DG ON


% ===============================
% CREATE Q NETWORK (DQN)
% ===============================

statePath = [

featureInputLayer(5,"Name","state")

fullyConnectedLayer(128,"Name","fc1")
reluLayer("Name","relu1")

fullyConnectedLayer(128,"Name","fc2")
reluLayer("Name","relu2")

fullyConnectedLayer(numel(actInfo.Elements),"Name","fc3")

];

criticNet = dlnetwork(layerGraph(statePath));


critic = rlVectorQValueFunction( ...
criticNet, ...
obsInfo, ...
actInfo, ...
ObservationInputNames = "state");


% ===============================
% AGENT OPTIONS
% ===============================

agentOpts = rlDQNAgentOptions(...
SampleTime = 1,...
UseDoubleDQN = true,...
TargetSmoothFactor = 1e-3,...
ExperienceBufferLength = 1e6,...
MiniBatchSize = 128);

agentOpts.EpsilonGreedyExploration.Epsilon = 1.0;

agentOpts.EpsilonGreedyExploration.EpsilonMin = 0.01;

agentOpts.EpsilonGreedyExploration.EpsilonDecay = 1e-4;

% ===============================
% CREATE AGENT
% ===============================

agentObj = rlDQNAgent(critic,agentOpts);


% ===============================
% EXPORT TO WORKSPACE
% ===============================

assignin("base","agentObj",agentObj);

disp("Microgrid EMS RL agentObj created successfully");

%% table of  rulebase output
% Access powers from Simulink output
Ppv   = out.rulePpv;
Pwind = out.rulePwind;
Pbatt = out.rulePbattery;
Pdg   = out.rulePdg;

% Create matching time vector for 24 sec simulation
N = length(Ppv);
t = linspace(0,24,N)';

% Required hourly points
hour = (0:24)';

% Interpolate powers at each hour
Ppv24   = interp1(t, Ppv, hour);
Pwind24 = interp1(t, Pwind, hour);
Pbatt24 = interp1(t, Pbatt, hour);
Pdg24   = interp1(t, Pdg, hour);

% Create table
T = table(hour, Ppv24, Pwind24, Pbatt24, Pdg24);

disp("RULEBASE OUTPUT");
disp(T)
%% rl output table
% Access powers from Simulink output
Ppv   = out.Ppv;
Pwind = out.Pwind;
Pbatt = out.Pbat;
Pdg   = out.Pdg;

% Create matching time vector for 24 sec simulation
N = length(Ppv);
t = linspace(0,24,N)';

% Required hourly points
hour = (0:24)';

% Interpolate powers at each hour
Ppv24   = interp1(t, Ppv, hour);
Pwind24 = interp1(t, Pwind, hour);
Pbatt24 = interp1(t, Pbatt, hour);
Pdg24   = interp1(t, Pdg, hour);

% Create table
T = table(hour, Ppv24, Pwind24, Pbatt24, Pdg24);

disp("RL OUTPUT");
disp(T)