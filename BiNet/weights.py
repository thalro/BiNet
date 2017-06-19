import numpy as np



def calc_js(Ns,ps,Ts,g):
    """ calculates j_ab so that J_ab = j_ab/\sqrt{N} is balanced and \sqrt{K} = p_ab*Nb excitatory
        spikes on average equal the threshold. 

        Ns:     List of population sizes
        
        ps:     2x2 matrix of connection probabilities

        Ts:     List of population thresholds

        g:      factor controlling relative strength of inhibition
        """
    Ns = np.array(Ns)
    ns = Ns/float(Ns.sum())
    js = np.zeros((2,2))
    js[0,0] = Ts[0] /np.sqrt(ps[0,0]*ns[0])
    js[0,1] = -g*js[0,0] *ns[0]*ps[0,0]/(ns[1]*ps[0,1])
    js[1,0] = Ts[1] /np.sqrt(ps[1,0]*ns[0])
    js[1,1] = -js[1,0] *ns[0]*ps[1,0]/(ns[1]*ps[1,1])
    return js


def generate_weight_block(N_out,N_in,p):
    """ generates a matrix with shape (N_out,N_in) whose entries
        are 1 with probability p an zero with probability (1-p).
        """
    return 1.*(np.random.rand(N_out,N_in)<p)
def generate_weight_matrix(Ns,ps,js,delta_j):
    """ generates a weight matrix composed of len(Ns)**2 blocks.
        blocks[i,j] have density ps[i,j] and non zero values
        js[i,j]/sqrt(Ns.sum()). if delta_j is nonzero or an array
        of shape (len(Ns),len(Ns)), the weights are drawn uniformly
        from [js[i,j]*(1-0.5*delta_j[i,j]),js[i,j]*(1+0.5*delta_j[i,j])].
        """
    Ns = np.array(Ns)
    Npop = len(Ns)
    N = Ns.sum()
    if delta_j is None:
        delta_j = np.zeros(ps.shape) 

    elif np.isscalar(delta_j):
        delta_j = np.ones_like(ps)*delta_j

    weights = []
    
    for i in range(Npop):
        weights.append([])
        for j in range(Npop):
            w = generate_weight_block(Ns[i],Ns[j],ps[i,j])
            w *= js[i,j]+delta_j[i,j]*js[i,j]* (np.random.rand(*w.shape)- 0.5)
            weights[-1].append(w)

    weights = np.concatenate([np.concatenate(w,axis=1) for w in weights],axis=0)
    weights[range(N),range(N)] *= 0.

    return weights/np.sqrt(N)

def generate_balanced_weights(Ns,ps,Ts,delta_j=None,g=1):
    """ generates a balanced weight matrix where sector weights
        are calculated with calc_js. Ns must have length 2.
        """
    js = calc_js(Ns,ps,Ts,g)
    

    return generate_weight_matrix(Ns,ps,js,delta_j)

def mazzucato_cluster_specs(Ns,ps,Ts,taus,g,f,Q,j_plus,gamma,original_j_minus = True):
    js = calc_js(Ns,ps,Ts,g)
    if original_j_minus:
        j_minus  = 1-gamma*f*(j_plus-1)
    else:
        j_minus = (Q-j_plus)/float(Q-1)
    Npops = Q+2

    newNs = np.ones(Npops,dtype=int)
    newNs[:Q] = Ns[0]*f/Q
    newNs[Q]  = Ns[0]-Q*Ns[0]*f/Q
    newNs[-1] = Ns[1]
    
    newjs = np.ones((Npops,Npops))*js[0,0]
    newjs[:Q,:Q]*= j_minus
    newjs[range(Q),range(Q)] = js[0,0] * j_plus
    newjs[-1,:-1] = js[1,0]
    newjs[-1,-1]  = js[1,1]
    newjs[:-1,-1] = js[0,1]

    newps =  np.ones((Npops,Npops)) * ps[0,0]  
    newps[-1,:-1] = ps[1,0]
    newps[-1,-1]  = ps[1,1]
    newps[:-1,-1] = ps[0,1]

    newTs = np.array([Ts[0]] * (Npops-1) +[Ts[1]])
    newtaus = np.array([taus[0]] * (Npops-1) +[taus[1]])

    return newNs,newps,newjs,newTs,newtaus


def doiron_cluster_specs(Ns,ps,Ts,taus,g,Q,R_EE,j_plus):
    #assert Ns[0]%Q == 0, 'Ns[0] must be divisible by Q'
    js = calc_js(Ns,ps,Ts,g)
    N_in = Ns[0]/Q
    N_out = Ns[0]-N_in
    p_out = Ns[0]*ps[0,0]/float(N_out+R_EE*N_in)
    p_in  = R_EE * p_out

    Npops = Q+1

    newNs = np.ones(Npops,dtype=int)
    newNs[:Q] = Ns[0]/Q
    newNs[-1] = Ns[1]
    
    newjs = np.ones((Npops,Npops))*js[0,0]
    newjs[range(Q),range(Q)] = js[0,0] * j_plus
    newjs[-1,:-1] = js[1,0]
    newjs[-1,-1]  = js[1,1]
    newjs[:-1,-1] = js[0,1]

    newps =  np.ones((Npops,Npops)) * p_out
    newps[range(Q),range(Q)] = p_in
    newps[-1,:-1] = ps[1,0]
    newps[-1,-1]  = ps[1,1]
    newps[:-1,-1] = ps[0,1]

    newTs = np.array([Ts[0]] * (Npops-1) +[Ts[1]])
    newtaus = np.array([taus[0]] * (Npops-1) +[taus[1]])


    return newNs,newps,newjs,newTs,newtaus




def EI_jplus_cluster_specs(Ns,ps,Ts,taus,g,Q,jplus,jip_factor = None):
    """ weights between populations are scaled so that Q E-I population pairs 
        are themselves balanced networks. j_in = j*jplus"""

    if jip_factor is not None:
        jep = jplus[0,0]
        jplus[:,:] = 1 +(jep-1)*jip_factor
        jplus[0,0] = jep
    js = calc_js(Ns,ps,Ts,g)
    Ns=np.array(Ns)
    

    Npops = Q*2

    newNs = np.ones(Npops,dtype=int)
    newNs[:Q] = Ns[0]/Q
    newNs[Q:] = Ns[1]/Q

    # EE
    newps = np.ones((Npops,Npops))*ps[0,0]
    # EI 
    newps[:Q,Q:] = ps[0,1]
    # IE
    newps[Q:,:Q] = ps[1,0]
    # II
    newps[Q:,Q:] = ps[1,1]
    
    newjs =  np.ones((Npops,Npops))

    # EE
    newjs[:Q,:Q] = js[0,0]*(Q-jplus[0,0])/(Q-1)
    newjs[range(Q),range(Q)] =js[0,0]*jplus[0,0]
    # EI 
    newjs[:Q,Q:] =js[0,1]*(Q-jplus[0,1])/(Q-1)
    newjs[range(Q),Q+np.arange(Q)] =js[0,1]*jplus[0,1]
    # IE
    newjs[Q:,:Q] = js[1,0]*(Q-jplus[1,0])/(Q-1)
    newjs[Q+np.arange(Q),range(Q)] = js[1,0]*jplus[1,0]
    # II
    newjs[Q:,Q:] = js[1,1]*(Q-jplus[1,1])/(Q-1)
    newjs[Q+np.arange(Q),Q+np.arange(Q)] = js[1,1]*jplus[1,1]


    newTs = np.array([Ts[0]] * Q +[Ts[1]] * Q)
    newtaus = np.array([taus[0]] * Q +[taus[1]] * Q)
   
   

    return newNs,newps,newjs,newTs,newtaus







def EI_cluster_specs(Ns,ps,Ts,taus,g,Q,fs):
    """ weights between populations are scaled so that Q E-I population pairs 
        are themselves balanced networks. j- = f*j+"""


    js = calc_js(Ns,ps,Ts,g)
    Ns=np.array(Ns)
    

    Npops = Q*2

    newNs = np.ones(Npops,dtype=int)
    newNs[:Q] = Ns[0]/Q
    newNs[Q:] = Ns[1]/Q

    # EE
    newps = np.ones((Npops,Npops))*ps[0,0]
    # EI 
    newps[:Q,Q:] = ps[0,1]
    # IE
    newps[Q:,:Q] = ps[1,0]
    # II
    newps[Q:,Q:] = ps[1,1]
    
    newjs =  np.ones((Npops,Npops))

    # EE
    newjs[:Q,:Q] = fs[0,0]*js[0,0]*Q/float((1+fs[0,0]*(Q-1)))
    newjs[range(Q),range(Q)] =js[0,0]*Q/float((1+fs[0,0]*(Q-1)))
    # EI 
    newjs[:Q,Q:] =fs[0,1]*js[0,1]*Q/float((1+fs[0,1]*(Q-1)))
    newjs[range(Q),Q+np.arange(Q)] =js[0,1]*Q/float((1+fs[0,1]*(Q-1)))
    # IE
    newjs[Q:,:Q] = fs[1,0]*js[1,0]*Q/float((1+fs[1,0]*(Q-1)))
    newjs[Q+np.arange(Q),range(Q)] = js[1,0]*Q/float((1+fs[1,0]*(Q-1)))
    # II
    newjs[Q:,Q:] = fs[1,1]*js[1,1]*Q/float((1+fs[1,1]*(Q-1)))
    newjs[Q+np.arange(Q),Q+np.arange(Q)] = js[1,1]*Q/float((1+fs[1,1]*(Q-1)))


    newTs = np.array([Ts[0]] * Q +[Ts[1]] * Q)
    newtaus = np.array([taus[0]] * Q +[taus[1]] * Q)
   
   

    return newNs,newps,newjs,newTs,newtaus


def BBN_cluster_specs(Ns,ps,Ts,taus,g,Q,fs):
    """ weights between populations are scaled so that Q E-I population pairs 
        are themselves balanced networks. j- = f*j+"""


    js = calc_js(Ns,ps,Ts,g)
    Ns=np.array(Ns)
    

    Npops = Q*2

    newNs = np.ones(Npops,dtype=int)
    newNs[:Q] = Ns[0]/Q
    newNs[Q:] = Ns[1]/Q

    # EE
    newps = np.ones((Npops,Npops))*ps[0,0]
    # EI 
    newps[:Q,Q:] = ps[0,1]
    # IE
    newps[Q:,:Q] = ps[1,0]
    # II
    newps[Q:,Q:] = ps[1,1]
    
    newjs =  np.ones((Npops,Npops))

    # EE
    newjs[:Q,:Q] = fs[0,0]*js[0,0]*Q**0.5
    newjs[range(Q),range(Q)] =js[0,0]*Q**0.5
    # EI 
    newjs[:Q,Q:] =fs[0,1]*js[0,1]*Q**0.5
    newjs[range(Q),Q+np.arange(Q)] =js[0,1]*Q**0.5
    # IE
    newjs[Q:,:Q] = fs[1,0]*js[1,0]*Q**0.5
    newjs[Q+np.arange(Q),range(Q)] = js[1,0]*Q**0.5
    # II
    newjs[Q:,Q:] = fs[1,1]*js[1,1]*Q**0.5
    newjs[Q+np.arange(Q),Q+np.arange(Q)] = js[1,1]*Q**0.5


    newTs = np.array([Ts[0]] * Q +[Ts[1]] * Q)
    newtaus = np.array([taus[0]] * Q +[taus[1]] * Q)
   
   

    return newNs,newps,newjs,newTs,newtaus





    
        







