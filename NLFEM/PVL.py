# -*- coding: utf-8 -*-

"""
Created on Sat Jan 25 18:38:22 2025

@author: KartiK
"""

import numpy as np
import matplotlib.pyplot as plt

# Input Parameters (Variant 0 from Table)

E, yo, eta, m = 80000, 160, 50, 1  # Material properties
A1, A2, L1, L2 = 8, 16, 40, 80    # Geometry
F_m, t_tot,  maxit, tol = 4200, 0.01, 10, 0.005  # Load, time, 
t_s = 4200 # Is the time step size 
h = t_tot / t_s  # deta t or time increment
t = np.arange(0, t_tot + h, h)  # Time array
F=(F_m/t_tot)*t  #load strored as per time step
# Nodes and Elements
n1 = 5
n2 = 5
n = n1 + n2 + 1  # Total nodes
l1, l2 = L1 / n1, L2 / n2  # Element lengths

# Storing and Post processing 
u_p1=[]
sigma_tb1=[]
strain_tb1=[]
sigma_tb2=[]
strain_tb2=[]


# Material Routine to update material parameter as yield condition
def material(strain,dt,ep):
    """
    Parameters
    ----------
    strain : total strain.
    dt : time increment
    ep : Initial plastic strain.

    Returns
    -------
    Ct : Material Stiffness.
    sigma : Stress .
    Ep : Updated plastic strain.

    """
    sigma_t=E*(strain-ep)
    lam=max(0, ((abs(sigma_t)/yo)-1))**m
    if lam==0:
        Ct=E
        sigma=sigma_t
        Ep=ep
        #print("Tangent Stiffness",Ct)
    else:
        
        Sigma_1 = ((sigma_t + E*h*eta*np.sign(sigma_t)) * yo )/ (yo + (E * h * eta))
        lamda=max(0, ((abs(Sigma_1)/yo)-1))**m
        Ep = ep + h*eta*lamda*np.sign(Sigma_1)
        sigma=Sigma_1
        #print("Sigma_1",sigma)
        Ct = E*yo/( yo + h*eta*E )
        
    return Ct, sigma,Ep


def elementStiffness( u_e, x, A,dt,ep):
    """

    Parameters
    ----------
    u_e : Elemental displacement for each element      
    x :  position of the elements 
    A : Cross section Area of the element
    dt : time step or time increment 
    ep : Initial Plastic strain

    Returns
    -------
    ke : Elemental Stiffeness matrix
    Fe_int : Internal Force for each element
    strain : Total strain for each element.
    sigma : Stress on each element
    E_p : Updated plastic strain .

    """
    # Gauss points and weights for 2-point Gauss quadrature
    xi = np.array([-np.sqrt(1 / 3), np.sqrt(1 / 3)])
    
    w = np.array([1, 1])  # Weights for each Gauss point
    ke = np.zeros((2, 2))  # Element stiffness matrix
    Fe_int = np.zeros(2)  # Internal force vector
    for i in range(len(xi)):
        # Shape functions and their derivatives
        N = np.array([0.5 * (1 - xi[i]), 0.5 * (1 + xi[i])])
        dN_dxi = np.array([-0.5, 0.5])
        J = np.dot(dN_dxi, x)
        B = dN_dxi / J  # Derivative of shape functions w.r.t physical coordinates
        strain = np.dot(B,u_e)
        
        Ct, sigma, E_p = material(strain,dt,ep)  # Material function call 
        
        ke += w[i] * Ct * A * np.outer(B, B) * J
        
        # Internal force vector contribution
        Fe_int += w[i] * sigma * A * B * J #  stress contribution for Fe_int
        
    return ke, Fe_int, strain, sigma, E_p


def main():
    """
    Implements the Newton Raphson Scheme.
    Assembly of elements, Boundary condition, and plotting are done in this 
    function.

    """
    itr=0
    u=np.zeros(n)
    Fext=np.zeros(n)
    ep_1=0
    ep_2=0
    count=0
    count1=0
    for itr in range(t_s+1):
        du=0
        Fext[n1]=F[itr]
        
        for i in range(maxit):#NR loop
            
            
            #initialize internal forces and residue at each NR iteration
            K=np.zeros([n,n])
            Fint=np.zeros([n])
            R=np.zeros(n)
            for k in range(n-1):# Loop for assembly
            
                if k < n1:  # Bar 1
                    L, A, ep = l1, A1, ep_1
                else:  # Bar 2
                    L, A, ep = l2, A2, ep_2
                
                x=np.array([k*L,(k+1)*L])  # position of each element
                Fe_int = 0  # Assigning a zero value initial to calculate elemental internal forces
                Ke_t = 0 # Assigning a zero value initial to calculate elemental tangent stiffness matrix
                Ae = np.zeros((2, n)) # Assignment matrix
                Ae[0][k] = 1  
                Ae[1][k+1] = 1 
                ue = np.matmul(Ae, u)  #Displacement for each element
            
                
                Ke_t,Fe_int,strain,sigma,Ep_pl = elementStiffness(ue,x,A,h,ep) #call of element routine
                
                
                if k < n1:
                    if Ep_pl!=0:
                            
                        if count==0: # To print when the plastic initiation takes place for bar 1
                            print("Plastic initiation of Bar 1 at laod:",F[itr])
                            count+=1
                        ep_1=Ep_pl
                    
                    stress_b1=sigma
                    strain_b1=strain
                    
                    
                else:
                    if Ep_pl !=0:
                        
                        if count1==0: # To print when the plastic initiation takes place for bar 1
                            print("Plastic initiation of Bar 2 at laod:",F[itr]) 
                            count1+=1
                        ep_2=Ep_pl
                    
                    
                    stress_b2=sigma
                    strain_b2=strain
                
                B = np.matmul(np.transpose(Ae), Ke_t) # Used for simplified matrix multiplication (Kint_inner)
                K += np.matmul(B,Ae)  
                Fint +=np.matmul(np.transpose(Ae),Fe_int)
                
                #  End of Assembly loop

            # Boundary condition indices (Dirichlet)
            # Eliminating  rows and columns as boundary conditions
            
            K_red1 = np.delete(np.delete(K,0, axis=0), 0, axis=1)# Remove row and column
            K_red = np.delete(np.delete(K_red1,n-2, axis=0), n-2, axis=1)
            Fext_red1 = np.delete(Fext, 0)  # Remove corresponding entry in Fext
            Fext_red = np.delete(Fext_red1, n-2) 
            u_red1 = np.delete(u, 0)        # Remove corresponding entry in displacement
            u_red = np.delete(u_red1, n-2)
            R_red1=np.delete(R,0)
            R_red=np.delete(R_red1,n-2)
            Fint1=np.delete(Fint,0)
            Fint_red=np.delete(Fint1,n-2)
            R_red = Fint_red -Fext_red
            
            # Calculation of Displacement change
            du = np.linalg.solve(K_red, -R_red)
            
            
            #Updating displacement
            u_red += du
            j=0
            for i in range(n):
                if i==0 or i==n-1:
                    u[i]= 0
                else:
                    u[i]=u_red[j]
                    j+=1      
            
            Fint_norm_m=np.linalg.norm(Fint,np.inf)
            R_max=np.linalg.norm(R_red,np.inf)
            du_norm_m = np.linalg.norm(du,np.inf) 
            u_norm_m = np.linalg.norm(u,np.inf) 
            
    
            if R_max <= tol*Fint_norm_m or du_norm_m <= tol*u_norm_m:
                #print("Converged!")
                break
        
        sigma_tb1.append(stress_b1) 
        strain_tb1.append(strain_b1)
        sigma_tb2.append(abs(stress_b2))
        strain_tb2.append(abs(strain_b2))
        u_p1.append(u[n1])
        
        if F[itr]== 2100:
            print("Dispalcemnt at Fext=2100",u_p1[itr])
    
   
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Stress-Strain Plot for Bar 1
    axs[0, 0].plot(strain_tb1, sigma_tb1,'b',lw=3, label='Bar 1')
    #axs[0, 0].plot(strain_tb1_p, sigma_tb1_p, label='Bar 1')
    axs[0, 0].set_title("Stress vs Strain - Bar 1")
    axs[0, 0].set_xlabel("Strain")
    axs[0, 0].set_ylabel("Stress")
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Stress-Strain Plot for Bar 2
    axs[0, 1].plot(strain_tb2, sigma_tb2,'g',lw=3, label='Bar 2')
    #axs[0, 1].plot(strain_tb2_p, sigma_tb2_p, label='Bar 2')
    axs[0, 1].set_title("Stress vs Strain - Bar 2")
    axs[0, 1].set_xlabel("Strain")
    axs[0, 1].set_ylabel("Stress in Mpa")
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # Force-Displacement Plot
    axs[1, 0].plot(u_p1, F,'r',lw=3,label='F')
    axs[1, 0].set_title("Force vs Displacement")
    axs[1, 0].set_xlabel("Displacement in mm")
    axs[1, 0].set_ylabel("Force in N")
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # Displacement-Time Plot
    axs[1, 1].plot(t, u_p1, lw=3,label='u')
    axs[1, 1].set_title("Displacement vs Time")
    axs[1, 1].set_xlabel("Time in s")
    axs[1, 1].set_ylabel("Displacement in mm")
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()







