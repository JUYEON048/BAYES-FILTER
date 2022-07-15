
xold = []
xestold
covestold

for t in time : # [0~0.01~20]

    # 1. real traj.

    xnew = f(xold,np.random()) # noise ~ N(0,Q)
    znew = h(xnew, np.random()) # noise ~ N(0,R)

    # 2 EFK

    xhat = f(xestold,0)
    covhat = F*covestold*F.T * Q

    H = h_jaco(xhat)
    K = H*covhat * np.inv(H*covhat*H.T+R)

    xestnew = xhat + K*(znew - h(xhat))
    covestnew = (np.eye(6)-K*H)*covhat




def f(xold, noise) :

    xnew = F*xold + G*u + noise
    return xnew

def h(x, noise) :
    return np.sqrt(x[0]^2+x[3]^2)

def h_jaco(x) :
    return np.array([x[0]/np.sqrt(x[0]^2+x[3]^2), 0, 0, ]