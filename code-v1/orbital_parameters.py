
def kepler_to_state(pb, ecc, a, 
        inclination, lan, 
        arg_periapsis, true_anomaly,
        mu):
    pass

def state_to_kepler(pos, vel, mu):
    ang_mom = np.cross(pos, vel) # in orbital plane
    ecc_vec = np.cross(vel, ang_mom)/mu - pos/np.sqrt(np.dot(pos,pos)) # points to periapsis
    p = np.dot(ang_mom, ang_mom)
    ecc = np.sqrt(np.dot(ecc_vec, ecc_vec))
    a = p/(1-e**2)
    
