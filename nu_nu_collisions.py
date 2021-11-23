
# coding: utf-8

# In[1]:


import numpy as np

from numba import jit, prange

GF=1.166*10**-11  #MeV**-2

# In[2]:


@jit(nopython=True)
def J1(p1,p2,p3):
    return ((16/15)*p3**3*(10*(p1+p2)**2-15*p3*(p1+p2)+6*p3**2))

@jit(nopython=True)
def J2(p1,p2):
    return ((16/15)*p2**3*(10*p1**2+5*p1*p2+p2**2))

@jit(nopython=True)
def J3(p1,p2,p3):
    return ((16/15)*((p1+p2)**5-10*p3**3*(p1+p2)**2+15*p3**4*(p1+p2)-6*p3**5))



# In[3]:


@jit(nopython=True)
def B1(i,j,f,p,dp):
    if i+j>=len(f)-1:
        u=len(f)-1
        h=(i+j)-u
    else:
        u=i+j
        h=0
    k=np.zeros(j+1-h)
    w=0
    for q in range(h,j+1):
        k[w]=q
        w=w+1
    FpJ1 = 0
    FmJ1 = 0
    for q in range(len(k)):
        FpJ1 = FpJ1 + (2*((1-f[i])*(1-f[j])*f[int(k[q])]*f[int(i+j-k[q])]*J1(p[i],p[j],p[int(k[q])])))
        FmJ1 = FmJ1 + (2*(f[i]*f[j]*(1-f[int(k[q])])*(1-f[int(i+j-k[q])])*J1(p[i],p[j],p[int(k[q])])))
    FpJ1i = (1-f[i])*(1-f[j])*f[h]*f[u]*J1(p[i],p[j],p[h])
    FpJ1f = (1-f[i])*(1-f[j])*f[j]*f[i]*J1(p[i],p[j],p[j])
    FmJ1i = f[i]*f[j]*(1-f[h])*(1-f[u])*J1(p[i],p[j],p[h])
    FmJ1f = f[i]*f[j]*(1-f[j])*(1-f[i])*J1(p[i],p[j],p[j])
    BP = (dp/2)*(FpJ1-FpJ1i-FpJ1f)
    BN = (dp/2)*(FmJ1-FmJ1i-FmJ1f)
    return BP,BN

@jit(nopython=True)
def A1(i,f,p,dp):
    BP=np.zeros(i+1)
    BN=np.zeros(i+1)
    for j in range(i+1):
        BP[j],BN[j]=B1(i,j,f,p,dp)
    AP=(dp/2)*(np.sum(2*BP)-BP[0]-BP[-1])
    AN=(dp/2)*(np.sum(2*BN)-BN[0]-BN[-1])
    return AP,AN

@jit(nopython=True)
def B2(i,j,f,p,dp):
    #if j>=i: m=i, n=j+1, o=-1
    #else: m=j, n=i+1, o=1
    if i+j>=len(f)-1:
        u=len(f)-1
        h=(i+j)-u
    else:
        u=i+j
        h=0
    #k=np.zeros(n-m)
    k = np.zeros(i+1-j)
    w=0
    #for q in range(m,n):
    for q in range(j,i+1):
        k[w]=q
        w=w+1
    FpJ2 = 0
    FmJ2 = 0
    for q in range(len(k)):
        FpJ2 = FpJ2 + (2*((1-f[i])*(1-f[j])*f[int(k[q])]*f[int(i+j-k[q])]*J2(p[i],p[j])))
        FmJ2 = FmJ2 + (2*(f[i]*f[j]*(1-f[int(k[q])])*(1-f[int(i+j-k[q])])*J2(p[i],p[j])))
    FpJ2i = (1-f[i])*(1-f[j])*f[i]*f[j]*J2(p[i],p[j])
    FpJ2f = (1-f[i])*(1-f[j])*f[j]*f[i]*J2(p[i],p[j])
    FmJ2i = f[i]*f[j]*(1-f[i])*(1-f[j])*J2(p[i],p[j])
    FmJ2f = f[i]*f[j]*(1-f[j])*(1-f[i])*J2(p[i],p[j])
    BP = (dp/2)*(FpJ2-FpJ2i-FpJ2f) #*o
    BN = (dp/2)*(FmJ2-FmJ2i-FmJ2f) #*o
    return BP,BN

@jit(nopython=True)
def A2(i,f,p,dp):
    BP=np.zeros(i+1)
    BN=np.zeros(i+1)
    for j in range(i+1): 
        BP[j],BN[j]=B2(i,j,f,p,dp)
    AP=(dp/2)*(np.sum(2*BP)-BP[0]-BP[-1])
    AN=(dp/2)*(np.sum(2*BN)-BN[0]-BN[-1])
    return AP,AN

@jit(nopython=True)
def B3(i,j,f,p,dp):
    if i+j>=len(f)-1:
        u=len(f)-1
        h=(i+j)-u
    else:
        u=i+j
        h=0
    k=np.zeros(u+1-i)
    w=0
    for q in range(i,u+1):
        k[w]=q
        w=w+1
    FpJ3 = 0
    FmJ3 = 0
    for q in range(len(k)):
        FpJ3 = FpJ3 + (2*((1-f[i])*(1-f[j])*f[int(k[q])]*f[int(i+j-k[q])]*J3(p[i],p[j],p[int(k[q])])))
        FmJ3 = FmJ3 + (2*(f[i]*f[j]*(1-f[int(k[q])])*(1-f[int(i+j-k[q])])*J3(p[i],p[j],p[int(k[q])])))
    FpJ3i = (1-f[i])*(1-f[j])*f[i]*f[j]*J3(p[i],p[j],p[i])
    FpJ3f = (1-f[i])*(1-f[j])*f[u]*f[h]*J3(p[i],p[j],p[u])
    FmJ3i = f[i]*f[j]*(1-f[i])*(1-f[j])*J3(p[i],p[j],p[i])
    FmJ3f = f[i]*f[j]*(1-f[u])*(1-f[h])*J3(p[i],p[j],p[u])
    BP = (dp/2)*(FpJ3-FpJ3i-FpJ3f)
    BN = (dp/2)*(FmJ3-FmJ3i-FmJ3f)
    return BP,BN

@jit(nopython=True)
def A3(i,f,p,dp):
    BP=np.zeros(i+1)
    BN=np.zeros(i+1)
    for j in range(i+1): #doesn't i get included twice, in both A1-3 and in A4-6?
        BP[j],BN[j]=B3(i,j,f,p,dp)
    AP=(dp/2)*(np.sum(2*BP)-BP[0]-BP[-1])
    AN=(dp/2)*(np.sum(2*BN)-BN[0]-BN[-1])
    return AP,AN

@jit(nopython=True)
def B4(i,j,f,p,dp):
    if i+j>=len(f)-1:
        u=len(f)-1
        h=(i+j)-u
    else:
        u=i+j
        h=0
    k=np.zeros(i+1-h)
    w=0
    for q in range(h,i+1):
        k[w]=q
        w=w+1
    FpJ1 = 0
    FmJ1 = 0
    for q in range(len(k)):
        FpJ1 = FpJ1 +(2*((1-f[i])*(1-f[j])*f[int(k[q])]*f[int(i+j-k[q])]*J1(p[i],p[j],p[int(k[q])])))
        FmJ1 = FmJ1 +(2*(f[i]*f[j]*(1-f[int(k[q])])*(1-f[int(i+j-k[q])])*J1(p[i],p[j],p[int(k[q])])))
    FpJ1i = (1-f[i])*(1-f[j])*f[h]*f[u]*J1(p[i],p[j],p[h])
    FpJ1f = (1-f[i])*(1-f[j])*f[i]*f[j]*J1(p[i],p[j],p[i])
    FmJ1i = f[i]*f[j]*(1-f[h])*(1-f[u])*J1(p[i],p[j],p[h])
    FmJ1f = f[i]*f[j]*(1-f[i])*(1-f[j])*J1(p[i],p[j],p[i])
    BP = (dp/2)*(FpJ1-FpJ1i-FpJ1f)
    BN = (dp/2)*(FmJ1-FmJ1i-FmJ1f)
    return BP,BN

@jit(nopython=True)
def A4(i,f,p,dp):
    BP=np.zeros(len(f)-1-i)
    BN=np.zeros(len(f)-1-i)
    v=0
    for j in range(i,len(f)-1): #why len(f)-1? Also doesn't i get included twice, in both A1-3 and in A4-6?
        BP[v],BN[v]=B4(i,j,f,p,dp)
        v=v+1
    AP=(dp/2)*(np.sum(2*BP)-BP[0]-BP[-1])
    AN=(dp/2)*(np.sum(2*BN)-BN[0]-BN[-1])
    return AP,AN

@jit(nopython=True)
def B5(i,j,f,p,dp):
    #if i>=j: m=j, n=i+1, o=-1
    #else: m=i, n=j+1, o=1
    if i+j>=len(f)-1:
        u=len(f)-1
        h=(i+j)-u
    else:
        u=i+j
        h=0
    #k=np.zeros(n-m)
    k = np.zeros(j+1-i)
    w=0
    #for q in range(m,n):
    for q in range(i,j+1):
        k[w]=q
        w=w+1
    FpJ2 = 0
    FmJ2 = 0
    for q in range(len(k)):
        FpJ2 = FpJ2 + (2*((1-f[i])*(1-f[j])*f[int(k[q])]*f[int(i+j-k[q])]*J2(p[j],p[i])))
        FmJ2 = FmJ2 + (2*(f[i]*f[j]*(1-f[int(k[q])])*(1-f[int(i+j-k[q])])*J2(p[j],p[i])))
    FpJ2i = (1-f[i])*(1-f[j])*f[i]*f[j]*J2(p[j],p[i])
    FpJ2f = (1-f[i])*(1-f[j])*f[j]*f[i]*J2(p[j],p[i])
    FmJ2i = f[i]*f[j]*(1-f[i])*(1-f[j])*J2(p[j],p[i])
    FmJ2f = f[i]*f[j]*(1-f[j])*(1-f[i])*J2(p[j],p[i])
    BP = (dp/2)*(FpJ2-FpJ2i-FpJ2f) #*o
    BN = (dp/2)*(FmJ2-FmJ2i-FmJ2f) #*o
    return BP,BN

@jit(nopython=True)
def A5(i,f,p,dp):
    BP=np.zeros(len(f)-1-i)
    BN=np.zeros(len(f)-1-i)
    v=0
    for j in range(i,len(f)-1):
        BP[v],BN[v]=B5(i,j,f,p,dp)
        v=v+1
    AP=(dp/2)*(np.sum(2*BP)-BP[0]-BP[-1])
    AN=(dp/2)*(np.sum(2*BN)-BN[0]-BN[-1])
    return AP,AN

@jit(nopython=True)
def B6(i,j,f,p,dp):
    if i+j>=len(f)-1:
        u=len(f)-1
        h=(i+j)-u
    else:
        u=i+j
        h=0
    k=np.zeros(u+1-j)
    w=0
    for q in range(j,u+1):
        k[w]=q
        w=w+1
    FpJ3 = 0
    FmJ3 = 0
    for q in range(len(k)):
        FpJ3 = FpJ3 + (2*((1-f[i])*(1-f[j])*f[int(k[q])]*f[int(i+j-k[q])]*J3(p[i],p[j],p[int(k[q])])))
        FmJ3 = FmJ3 + (2*(f[i]*f[j]*(1-f[int(k[q])])*(1-f[int(i+j-k[q])])*J3(p[i],p[j],p[int(k[q])])))
    FpJ3i = (1-f[i])*(1-f[j])*f[j]*f[i]*J3(p[i],p[j],p[j])
    FpJ3f = (1-f[i])*(1-f[j])*f[u]*f[h]*J3(p[i],p[j],p[u])
    FmJ3i = f[i]*f[j]*(1-f[j])*(1-f[i])*J3(p[i],p[j],p[j])
    FmJ3f = f[i]*f[j]*(1-f[u])*(1-f[h])*J3(p[i],p[j],p[u])
    BP = (dp/2)*(FpJ3-FpJ3i-FpJ3f)
    BN = (dp/2)*(FmJ3-FmJ3i-FmJ3f)
    return BP,BN

@jit(nopython=True)
def A6(i,f,p,dp):
    BP=np.zeros(len(f)-1-i)
    BN=np.zeros(len(f)-1-i)
    v=0
    for j in range(i,len(f)-1):
        BP[v],BN[v]=B6(i,j,f,p,dp)
        v=v+1
    AP=(dp/2)*(np.sum(2*BP)-BP[0]-BP[-1])
    AN=(dp/2)*(np.sum(2*BN)-BN[0]-BN[-1])
    return AP,AN


# In[4]:


@jit(nopython=True)
def cI(i,f,p):
    dp=p[1]-p[0]
    coefficient = GF**2/((2*np.pi)**3*p[i]**2)
    AP1,AN1=A1(i,f,p,dp)
    AP2,AN2=A2(i,f,p,dp)
    AP3,AN3=A3(i,f,p,dp)
    AP4,AN4=A4(i,f,p,dp)
    AP5,AN5=A5(i,f,p,dp)
    AP6,AN6=A6(i,f,p,dp)
    c=coefficient*((AP1-AN1)+(AP2-AN2)+(AP3-AN3)+(AP4-AN4)+(AP5-AN5)+(AP6-AN6))
    FRS=coefficient*((AP1+AN1)+(AP2+AN2)+(AP3+AN3)+(AP4+AN4)+(AP5+AN5)+(AP6+AN6))
    return c,FRS

@jit(nopython=True,parallel=True)
def C(p,f):
    c=np.zeros(len(p))
    FRS=np.zeros(len(p))
    for i in prange(1,len(p)-1): #i only goes up to 199 because in a# def's goes i-200 nd if i is 200 then len0
        c[i],FRS[i]=cI(i,f,p)
        if (np.abs(c[i])/FRS[i])<=3e-15:
            c[i]=0
    return c


@jit(nopython=True)
def C_nopar(p,f):
    c=np.zeros(len(p))
    FRS=np.zeros(len(p))
    for i in prange(1,len(p)-1): #i only goes up to 199 because in a# def's goes i-200 nd if i is 200 then len0
        c[i],FRS[i]=cI(i,f,p)
        if (np.abs(c[i])/FRS[i])<=3e-15:
            c[i]=0
    return c
