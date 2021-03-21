function [x1c, x2c, x3c, param] = GissingerControl(period)

% Function for stabilizing UPOs in the Gissinger flow through perturbations 
% in the mu parameter. This code is associated to the work of Section 4.3 
% of Deep Learning of Conjugate Mappings by Jason J. Bramburger, Steven L. 
% Brunton, and J. Nathan Kutz (2021). UPOs are stabilized using the work 
% of Section 2.4, based on the work in Data-Driven Stabilization of 
% Periodic Orbits by Bramburger, Kutz, and Brunton (2021).
%
% Inputs:   - Period of the orbit to stabilize 
%               - Acceptable sequences period = 1,2,3,4 
%               - Default period is 1
%               - Invalid inputs are changed to the default
%               - There are two distinct period 2 orbits. The user can 
%                   comment and uncomment in the options below to select
%                   between them
% 
% Outputs:  - x1c: the controlled x1 component
%           - x2c: the controlled x2 component
%           - x3c: the controlled x3 component
%           - param: mu values used to control the orbit (mu_n in manuscript) 
%

% Model parameters 
nu = 0.1;
gam = 0.85;
mu = 0.12; % focal parameter value

% Integration parameters
dt = 0.001;
tspan = 0:dt:50;
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,3));

% Set default input
if nargin == 0 
    fprintf('No period given. Period set to default value of 1. \n')
    period = 1;
elseif mod(period,1) > 0 
    fprintf('Non-integer periods are not accepted. Period set to 1. \n')
    period = 1;
elseif period > 4
    fprintf('Periods larger than 6 are not accepted. Period set to 6. \n')
    period = 4;
elseif period < 1
    fprintf('Periods smaller than 1 are not accepted. Period set to 1. \n')
    period = 1;
end

% Get UPO values
if period == 1
   % Period 1 specs 
   x2UPO = 1.2179209;
   x3UPO = 1.3286662;
   K1 = -0.3145;
   K2 = 0.0012;
   eta = 0.05;
elseif period == 2 % First period 2 orbit
   % Period 2 specs 
   x2UPO = [1.0727983; 1.2868291];
   x3UPO = [1.1192245; 1.4253796];
   K1 = [0; -0.652786422593964];
   K2 = [0; -0.020396202569044];
   eta = 0.05;
% elseif period == 2 % Second period 2 orbit
%    % Period 2 specs 
%    x2UPO = [1.1593777; 1.43768131];
%    x3UPO = [1.247489; 1.66258233];
%    K1 = [-0.097506200071231; 0.379934437228873];
%    K2 = [0.015378083516764; 0.038812073048770];
%    eta = 0.05;
elseif period == 3 
   % Period 3 specs 
   x2UPO = [1.0502075; 1.1965837; 1.2992355];
   x3UPO = [1.0815638; 1.2989657; 1.4421626];
   K1 = [0.0; -0.209553614684828; -0.745931146626367];
   K2 = [0.0; 0.005363056398677; -0.025813817098888];
   eta = 0.01;
elseif period == 4 
   % Period 4 specs 
   x2UPO = [1.4952934; 1.20747; 1.2420436; 1.1428387];
   x3UPO = [1.6964461; 1.3139266; 1.3622724; 1.2215892];
   K1 = [0.047124701954694; -0.229738843454402; -0.400904594063539; 0.027650297128145];
   K2 = [0.010392535321082; 0.002672235172716; -0.004346347517524; 0.006085756709558];
   eta = 0.001;
end

% Threshold parameter (period 6 requires more precision)
% if period <= 5
%     eta = 0.5;
% else
%     eta = 0.1;
% end

% Initial condition close to unstable orbit
x0(1,:) = [-x2UPO(1); x2UPO(1); x3UPO(1)];

% Controlled parameter
if (x0(1,2) - x2UPO(1))^2 + (x0(1,3) - x3UPO(1))^2 <= eta 
    param(1) = mu + K1(1)*(x0(1,2) - x2UPO(1)) + K2(1)*(x0(1,3) - x3UPO(1));
else
    param(1) = mu;
end

% Initialize trajectory
[~,sol] = ode45(@(t,x) Gissinger(x,param(1),nu,gam),tspan,x0(1,:),options);

% Initialize Controlled Solution
x1c = [];
x2c = [];
x3c = [];

% Controlled orbit
kfinal = 50;
for k = 2:kfinal
    
    for j = 1:length(sol(:,1))-1
       if  ((sol(j,1) + sol(j,2)) < 0) && ((sol(j+1,1) + sol(j+1,2)) >= 0)  
%             if abs(sol(j,1) + sol(j,2)) <= abs(sol(j+1,1) + sol(j+1,2))
%                ind = j;
%             else
%                ind = j+1; 
%             end
            ind = j;
            
            % Controlled solution
            x1c = [x1c; sol(1:ind,1)];
            x2c = [x2c; sol(1:ind,2)];
            x3c = [x3c; sol(1:ind,3)];
            
            break
        end 
    end
   
    
    x0(k,:) = [-sol(ind,2); sol(ind,2); sol(ind,3)];
    if (x0(k,2) - x2UPO(1))^2 + (x0(k,3) - x3UPO(1))^2 <= eta 
        param(k) = mu + K1(1)*(x0(k,2) - x2UPO(1)) + K2(1)*(x0(k,3) - x3UPO(1));
    elseif length(x2UPO) >= 2 && (x0(k,2) - x2UPO(2))^2 + (x0(k,3) - x3UPO(2))^2 <= eta 
        param(k) = mu + K1(2)*(x0(k,2) - x2UPO(2)) + K2(2)*(x0(k,3) - x3UPO(2));
    elseif length(x2UPO) >= 3 && (x0(k,2) - x2UPO(3))^2 + (x0(k,3) - x3UPO(3))^2 <= eta 
        param(k) = mu + K1(3)*(x0(k,2) - x2UPO(3)) + K2(3)*(x0(k,3) - x3UPO(3));
    elseif length(x2UPO) >= 4 && (x0(k,2) - x2UPO(4))^2 + (x0(k,3) - x3UPO(4))^2 <= eta 
        param(k) = mu + K1(4)*(x0(k,2) - x2UPO(4)) + K2(4)*(x0(k,3) - x3UPO(4));
    else
        param(k) = mu;
    end
    
    [~,sol] = ode45(@(t,x) Gissinger(x,param(k),nu,gam),tspan,x0(k,:),options);
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

%% Gissinger right-hand-side

function dx = Gissinger(x,mu,nu,gamma)
    
    % Equilibria for scaling
    xstar = sqrt(nu + gamma*sqrt(nu/mu));
    ystar = sqrt(mu + gamma*sqrt(mu/nu));
    zstar = -sqrt(nu*mu) - gamma;
    
    % Rescaled variables
    x1hat = x(1)*xstar;
    x2hat = x(2)*ystar;
    x3hat = x(3)*zstar;

    dx = [(mu*x1hat - x2hat*(x3hat + gamma))/xstar; (-nu*x2hat + x1hat*(x3hat + gamma))/ystar; (-x3hat + x1hat*x2hat)/zstar];

end
