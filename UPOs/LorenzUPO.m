function [x1UPO, x2UPO, x3UPO, period] = LorenzUPO(sequence)

% Function for producing UPOs in the Lorenz flow. This code is associated 
% to the work of Section 4.2 of Deep Learning of Conjugate Mappings by 
% Jason J. Bramburger, Steven L. Brunton, and J. Nathan Kutz (2021). UPOs 
% are initialized using the method presented in Section 2.3
%
% Inputs:   - Sequences of L's and R's 
%               - Acceptable sequences are listed in Table 2 in Section 4.2
%               - Default sequence is LR
%               - Invalid inputs are changed to the default
% 
% Outputs:  - x1UPO: the x1 component of the UPO
%           - x2UPO: the x2 component of the UPO
%           - x3UPO: the x3 component of the UPO
%           - period: the temporal period of the UPO
%
% Note: the scaling term 0.1287787921511248 scales the x1 and x2 section
% data to be in the cube [-1,1]. This was used to train the network and
% therefore the initial conditions reflect this scaling.

% Model parameters 
sigma = 10;
beta = 8/3;
rho = 28;
scale = 0.1287787921511248; 

% Integration parameters
m = 3; %Dimension of ODE
dt = 0.0025;
tspan = 0:dt:50;
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,m));

% Set default input
if nargin == 0 
    sequence = 'LR';
end

% Initial conditions in Poincare section
if strcmp(sequence,'LR') == 1
    % LR periodic orbit
    x0(1,:) = [0.2534494;  -0.26987886; rho - 1];
    x0(2,:) = [-0.27675274;  0.2568418; rho - 1];
elseif strcmp(sequence,'LLR') == 1
    % LLR periodic orbit
    x0(1,:) = [0.48546383; 0.05931851; rho - 1];
    x0(2,:) = [0.31559905; -0.19357193; rho - 1];
    x0(3,:) = [-0.11501984; 0.46069524; rho - 1];
elseif strcmp(sequence,'LLRR') == 1
    % LLRR periodic orbit
    x0(1,:) = [0.41026473; -0.05038974; rho - 1];
    x0(2,:) = [0.15894957; -0.39100677; rho - 1];
    x0(3,:) = [-0.43584028; 0.03873325; rho - 1];
    x0(4,:) = [-0.18420194; 0.378088; rho - 1];
elseif strcmp(sequence,'LLLR') == 1
    % LLLR periodic orbit
    x0(1,:) = [0.5942744;   0.22637127; rho - 1];
    x0(2,:) = [0.5029391; 0.08536107; rho - 1];
    x0(3,:) = [0.33850104; -0.16484758; rho - 1];
    x0(4,:) = [-0.00682702; 0.57997245; rho - 1];
elseif strcmp(sequence,'LLLLR') == 1
    % LLLLR periodic orbit
    x0(1,:) = [0.66089654;  0.33113882; rho - 1];
    x0(2,:) = [0.6003475;   0.23560898; rho - 1];
    x0(3,:) = [0.5114844;   0.09871124; rho - 1];
    x0(4,:) = [0.35044855; -0.15020359; rho - 1];
    x0(5,:) = [0.06531161;  0.6611144; rho - 1];
elseif strcmp(sequence,'LLLRR') == 1
    % LLLRR periodic orbit
    x0(1,:) = [0.5356805;   0.13213904; rho - 1];
    x0(2,:) = [0.42002743; -0.03562031; rho - 1];
    x0(3,:) = [0.18623897; -0.35744023; rho - 1];
    x0(4,:) = [-0.4040897;   0.08395283; rho - 1];
    x0(5,:) = [-0.07085289;  0.5129924; rho - 1];
elseif strcmp(sequence,'LLRLR') == 1
    % LLRLR periodic orbit
    x0(1,:) = [0.4565648;   0.01834113; rho - 1];
    x0(2,:) = [0.2705094;  -0.2490433; rho - 1];
    x0(3,:) = [-0.24326596;  0.30084667; rho - 1];
    x0(4,:) = [0.3086903;  -0.20198391; rho - 1];
    x0(5,:) = [-0.13977657;  0.4315217; rho - 1];
else
    fprintf('Not an acceptable sequence. Sequence set to LR. \n')
    % Default periodic orbit
    x0(1,:) = [0.2534494;  -0.26987886; rho - 1];
    x0(2,:) = [-0.27675274;  0.2568418; rho - 1];
end

% Create initial guess
perMap = length(x0(:,1)); % period of UPO in Poincare map
xinit = [];
yinit = [];
zinit = [];

for p = 1:perMap

    % Initialize trajectory
    [~,sol] = ode45(@(t,x) Lorenz(x,sigma,beta,rho),tspan,x0(p,:),options);

    ind = 0;
    for j = 1:length(sol(:,1))-1
       if  (sol(j,3) > rho - 1) && (sol(j+1,3) <= rho - 1) % Poincare section  
            if abs(sol(j,3) - rho + 1) <= abs(sol(j+1,3) - rho + 1)
                ind = j;
            else
               ind = j+1; 
            end

            xinit = [xinit; sol(1:ind,1)];
            yinit = [yinit; sol(1:ind,2)];
            zinit = [zinit; sol(1:ind,3)];

            break
        end 
    end

end

% Root finding algorithm
tic % start timer

T = dt*length(xinit); % initial guess at the period
N = length(xinit);

% First Derivative
D = sparse(1:N-1,[2:N-1 N],ones(N-1,1)/2,N,N);
D(N,1) = 0.5; % Periodic BCs
D = (D - D')/dt;

init = [xinit; yinit; zinit; T]; % Initial guess

% fsolve options
options=optimset('Display','iter','Jacobian','on','MaxIter',10000,'Algorithm','levenberg-marquardt','TolFun',1e-15,'TolX',1e-15);

% call fsolve
xNew = fsolve(@(x) Periodic(x,D,init,sigma,beta,rho,N),init,options);

toc %end timer

% Recover solutions (flip x & y because the conjugacy flips them in
% latent map)
x1UPO = -xNew(1:N)/scale; 
x2UPO = -xNew(N+1:2*N)/scale;
x3UPO = xNew(2*N+1:3*N);
period = tspan(N)/xNew(end);

% Plot the solution in the x1-x3 plane
plot([-x1UPO; -x1UPO],[x3UPO; x3UPO],'Linewidth',5)
set(gca,'fontsize',16)
xlabel('$x(t)$','Interpreter','latex','FontSize',20,'FontWeight','Bold')
ylabel('$z(t)$','Interpreter','latex','FontSize',20,'FontWeight','Bold')
    
end

%% Lorenz right-hand-side

function dx = Lorenz(x,sigma,beta,rho)

    scale = 0.1287787921511248;

    dx = [sigma*(x(2) - x(1)); x(1)*(rho - x(3)) - x(2); (scale^(-2))*x(1)*x(2) - beta*x(3)];
    
end

%% Function whose roots are periodic orbits

function [F,J] = Periodic(xin,D,xinit,sigma,beta,rho,N)

  x = xin(1:N);
  y = xin(N+1:2*N);
  z = xin(2*N+1:3*N);
  T = xin(end);
  
  scale = 0.1287787921511248;

  % Right-hand side
  F(1:N) =  D*x - T*(sigma*(y - x));
  F(N+1:2*N) =  D*y - T*(x.*(rho - z) - y);
  F(2*N+1:3*N) =  D*z - T*((scale^(-2))*x.*y - beta*z);
  F(3*N+1) =  (dot(D*xinit(1:N),xinit(1:N)-x) + dot(D*xinit(N+1:2*N),xinit(N+1:2*N)-y) + dot(D*xinit(2*N+1:3*N),xinit(2*N+1:3*N)-z));

  % Jacobian
  if nargout > 1
      e = ones(N,1);
      
      J = sparse(3*N+1,3*N+1);
      
      % First component
      J(1:N,1:N) = D + spdiags(T*sigma*e, 0, N, N);
      J(1:N,N+1:2*N) = spdiags(-T*sigma*e, 0, N, N);
      J(1:N,3*N+1) = -sigma*(y - x); 
      
      % Second component 
      J(N+1:2*N,1:N) = spdiags(-T*(rho - z), 0, N, N);
      J(N+1:2*N,N+1:2*N) = D + spdiags(T*e, 0, N, N);
      J(N+1:2*N,2*N+1:3*N) = spdiags(T*x, 0, N, N); 
      J(N+1:2*N,3*N+1) = -(x.*(rho - z) - y); 
      
      % Third component
      J(2*N+1:3*N,1:N) = spdiags(-T*(scale^(-2))*y, 0, N, N);
      J(2*N+1:3*N,N+1:2*N) = spdiags(-T*(scale^(-2))*x, 0, N, N);
      J(2*N+1:3*N,2*N+1:3*N) = D + spdiags(T*beta*e, 0, N, N);
      J(2*N+1:3*N,3*N+1) = -((scale^(-2))*x.*y - beta*z);
      
      % Fourth component
      J(3*N+1,1:N) = -D*xinit(1:N);
      J(3*N+1,N+1:2*N) = -D*xinit(N+1:2*N);
      J(3*N+1,2*N+1:3*N) = -D*xinit(2*N+1:3*N);
  end

end