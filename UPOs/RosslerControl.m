function [x1c, x2c, x3c, param, A, B] = RosslerControl(period)

% Function for stabilizing UPOs in the Rossler flow. This code is associated 
% to the work of Section 4.1 of Deep Learning of Conjugate Mappings by 
% Jason J. Bramburger, Steven L. Brunton, and J. Nathan Kutz (2021). UPOs 
% are stabilized using the work of Section 2.4, based on the work in 
% Data-Driven Stabilization of Periodic Orbits by Bramburger, Kutz, and 
% Brunton (2021).
%
% Inputs:   - Period of the orbit to stabilize 
%               - Acceptable sequences period = 1,2,3,4,5,6
%               - Default period is 1
%               - Invalid inputs are changed to the default
% 
% Outputs:  - x1c: the controlled x1 component
%           - x2c: the controlled x2 component
%           - x3c: the controlled x3 component
%           - param: c values used to control the orbit (mu_n in manuscript) 
%           - A: Matrix containing the A_i values of the x2 derivative of
%               the Poincare map at each UPO and parameter value
%           - B: Matrix containing the B_i values of the c derivative of
%               the Poincare map at each UPO and parameter value
%

% Model parameters 
a = 0.1;
b = 0.1;
c = 11; % focal parameter value

% Integration parameters
dt = 0.001;
tspan = 0:dt:10;
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,3));

% Set default input
if nargin == 0 
    fprintf('No period given. Period set to default value of 1. \n')
    period = 1;
elseif mod(period,1) > 0 
    fprintf('Non-integer periods are not accepted. Period set to 1. \n')
    period = 1;
elseif period > 6
    fprintf('Periods larger than 6 are not accepted. Period set to 6. \n')
    period = 6;
elseif period < 1
    fprintf('Periods smaller than 1 are not accepted. Period set to 1. \n')
    period = 1;
end

% Get UPO values
if period == 1
   % Period 1 specs 
   x2UPO = -15.0865;
   A = -2.1775;
   B = -3.6047;
   K = -0.5;
elseif period == 2
   % Period 2 specs
   x2UPO = [-12.161534; -16.289286];
   A = [0.9748; -2.7329];
   B = [-0.4009; -4.1414];
   K = [0, -0.5];
elseif period == 3
   % Period 3 specs
   x2UPO = [-10.450764; -13.428555; -16.793669];
   A = [1.2896; 0.1536; -2.8130];
   B = [-0.0804; -1.2414; -4.1095];
   K = [0.0, 0.0, -0.75];
elseif period == 4
   % Period 4 specs
   x2UPO = [-11.535787; -15.564866; -14.081667; -16.470816];
   A = [2.3086; -2.5076; -0.9247; -2.7519];
   B = [-0.4355; -3.9826; -2.3458; -4.1408];
   K = [0.0, -0.5, 0.0, -0.5];
elseif period == 5
   % Period 5 specs
   x2UPO = [-10.941585; -14.747766; -15.658405; -13.871784; -16.622202];
   A = [1.2445; -1.8426; -2.5533; -0.6470; -2.7598];
   B = [-0.1260; -3.2746; -4.0213; -2.0614; -4.1311];
   K = [0.0, 0.0, -0.5, 0.0, -0.5];
elseif period == 6
   % Period 6 specs
   x2UPO = [-11.301012; -15.243899; -14.781998; -15.602014; -14.000036; -16.535553];
   A = [1.1967; -2.3047; -3.3120; -2.5263; -0.7333; -2.7561];
   B = [-0.1745; -3.7269; -1.8026; -3.9989; -2.2345; -4.1375];
   K = [0.0, 0.0, -0.5, 1.0, -0.5, -0.75];
end

% Threshold parameter (period 6 requires more precision)
if period <= 5
    eta = 0.5;
else
    eta = 0.1;
end

% Initial condition close to unstable orbit
x0(1,:) = [0; x2UPO(1); 0];

% Controlled parameter
if abs(x0(1,2) - x2UPO(1)) <= eta 
    param(1) = c + K(1)*(x0(1,2) - x2UPO(1));
else
    param(1) = c;
end

% Initialize trajectory
[~,sol] = ode45(@(t,x) Rossler(x,a,b,param(1)),tspan,x0(1,:),options);

% Initialize Controlled Solution
x1c = [];
x2c = [];
x3c = [];

% Controlled orbit
kfinal = 100;
for k = 2:kfinal
    
    for j = 1:length(sol(:,1))
       if  (sol(j,1) < 0) && (sol(j+1,1) >= 0)  
            if abs(sol(j,1)) <= abs(sol(j+1,2))
                ind = j;
            else
               ind = j+1; 
            end
            
            % Controlled solution
            x1c = [x1c; sol(1:ind,1)];
            x2c = [x2c; sol(1:ind,2)];
            x3c = [x3c; sol(1:ind,3)];
            
            break
        end 
    end
   
    
    x0(k,:) = [0; sol(ind,2); 0];
    if abs(x0(k,2) - x2UPO(1)) <= eta 
        param(k) = c + K(1)*(x0(k,2) - x2UPO(1));
    elseif length(x2UPO) >= 2 && abs(x0(k,2) - x2UPO(2)) <= eta 
        param(k) = c + K(2)*(x0(k,2) - x2UPO(2));
    elseif length(x2UPO) >= 3 && abs(x0(k,2) - x2UPO(3)) <= eta 
        param(k) = c + K(3)*(x0(k,2) - x2UPO(3));
    elseif length(x2UPO) >= 4 && abs(x0(k,2) - x2UPO(4)) <= eta 
        param(k) = c + K(4)*(x0(k,2) - x2UPO(4));
    elseif length(x2UPO) >= 5 && abs(x0(k,2) - x2UPO(5)) <= eta 
        param(k) = c + K(5)*(x0(k,2) - x2UPO(5));
    elseif length(x2UPO) >= 6 && abs(x0(k,2) - x2UPO(6)) <= eta 
        param(k) = c + K(6)*(x0(k,2) - x2UPO(6));
    else
        param(k) = c;
    end
    
    [~,sol] = ode45(@(t,x) Rossler(x,a,b,param(k)),tspan,x0(k,:),options);
end

% Last Iteration of Controlled solution
x1c = [x1c; sol(1:ind,1)];
x2c = [x2c; sol(1:ind,2)];
x3c = [x3c; sol(1:ind,3)];

% Plot Solutions
figure(1)
plot3(x1c(floor(end/2):end),x2c(floor(end/2):end),x3c(floor(end/2):end),'b','LineWidth',4) % eliminate transients
set(gca,'FontSize',16)
xlabel('$x_1(t)$','Interpreter','latex','FontSize',20,'FontWeight','Bold')
ylabel('$x_2(t)$','Interpreter','latex','FontSize',20,'FontWeight','Bold')
zlabel('$x_3(t)$','Interpreter','latex','FontSize',20,'FontWeight','Bold')
grid on

end

%% Rossler right-hand-side

function dx = Rossler(x,a,b,c)

    dx = [-x(2) - x(3); x(1) + a*x(2); b + x(3)*(x(1) - c)];

end