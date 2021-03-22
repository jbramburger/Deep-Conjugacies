# Rossler right-hand-side
def Rossler(x,t,c = 11):
    a = 0.1
    b = 0.1
    
    dxdt = [-x[1] - x[2], x[0] + a*x[1], b + x[2]*(x[0] - c)]
    return dxdt
 
 # Lorenz right-hand-side
def Lorenz(x,t,rho = 28.0):
    sig = 10.0 
    beta = 8.0/3.0
    
    dxdt = [sig*(x[1] - x[0]), x[0]*(rho - x[2]) - x[1], x[0]*x[1] - beta*x[2]]
    return dxdt   
    
# N mode Galerkin projection of Kuramoto-Shivashinsky PDE
def Kuramoto(x,t, nu = 0.0298, modes = 14):
    
    # Initialize
    dxdt = [0] * modes
    
    # Loop over modes
    for k in range(modes):
        
        dxdt[k] = ((k+1)**2)*(1 - nu*((k+1)**2))*x[k]
        
        if k > 0:
            for n in range(k):
                dxdt[k] = dxdt[k] - 0.25*(k+1)*x[n]*x[k-n-1]
                
        for m in range(modes - k - 1):
            dxdt[k] = dxdt[k] + 0.5*(k+1)*x[m]*x[k+m+1]
            
    return dxdt    
    

    