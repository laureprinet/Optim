##################### PROJET OPTIMISATION GROUPE 2 #####################################
import numpy as np
from casadi import Opti
import casadi as cs


### Question 6 ###

# Données :
alpha = 0.1
c = 10 ** (-3) * np.array([30, 1, 1.3, 4, 1])
v = np.array([0.9, 1.5, 1.1])
d = np.array([400, 67, 33])
A = np.array([[3.5, 2, 1], [250, 80, 25], [0, 8, 3], [0, 40, 10], [0, 8.5, 0]])


# Fonction h
def fun_h(a, q, d):
    h = (q * cs.exp(-a * q) + d * cs.exp(-a * d)) / (cs.exp(-a * q) + cs.exp(-a * d))
    return h


"""

# Définition des variables pour minimisation
opti_cons = Opti()
q = opti_cons.variable(3)
r = opti_cons.variable(5)

"""


"""
# Minimisation de -f(q,r)
opti_cons.minimize(- cs.dot(v, fun_h(alpha, q, d)) + cs.dot(c, r))
# Contraintes à respecter
opti_cons.subject_to(-r <= 0)
opti_cons.subject_to(-q <= 0)
opti_cons.subject_to(A @ q - r <= 0)


opti_cons.solver("ipopt")
print("Solve Chocolatine with constraints with Casadi...")
sol = opti_cons.solve()
print(f"q = {sol.value(q)}")
print(f"r = {sol.value(r)}")


"""

"""
### Question 7b ###

# Données supplémentaires
K = 3
pi = [0.5, 0.3, 0.2]
d = [np.array([400, 67, 33]), np.array([500, 80, 53]), np.array([300, 60, 43])]

# Minimisation de -f(q,r)
f = sum([pi[i] * (cs.dot(v, fun_h(alpha, q, d[i])) - cs.dot(c, r)) for i in range(3)])
opti_cons.minimize(-f)
# Contraintes à respecter
opti_cons.subject_to(-r <= 0)
opti_cons.subject_to(-q <= 0)
opti_cons.subject_to(A @ q - r <= 0)


opti_cons.solver("ipopt")
print("Solve Chocolatine with constraints with Casadi...")
sol = opti_cons.solve()
print(f"q = {sol.value(q)}")
print(f"r = {sol.value(r)}")

"""


### Question 9c ###

# Définition des variables pour minimisation
opti_cons = Opti()
q = opti_cons.variable(3)
u0 = opti_cons.variable(3)
u1 = opti_cons.variable(3)
u2 = opti_cons.variable(3)


# Données supplémentaires
K = 3
pi = [0.5, 0.3, 0.2]
c = [
    10 ** (-3) * np.array([30, 1, 1.3, 4, 1]),
    10 ** (-3) * np.array([40, 1.2, 1.2, 3, 1.2]),
    10 ** (-3) * np.array([20, 0.8, 1.4, 5, 0.8]),
]
d = np.array([400, 67, 33])
uk = [u0, u1, u2]


# Minimisation de -f(q,r)
f = 0
for i in range(3):
    f += pi[i] * (cs.dot(v, uk[i]) - cs.dot(c[i], A @ q))
opti_cons.minimize(-f)
# Contraintes à respecter
opti_cons.subject_to(uk[0] - q <= 0)
opti_cons.subject_to(uk[1] - q <= 0)
opti_cons.subject_to(uk[2] - q <= 0)
opti_cons.subject_to(uk[0] - d <= 0)
opti_cons.subject_to(uk[1] - d <= 0)
opti_cons.subject_to(uk[2] - d <= 0)
opti_cons.subject_to(-q <= 0)


opti_cons.solver("ipopt")
print("Solve Chocolatine with constraints with Casadi...")
sol = opti_cons.solve()
print(f"q = {sol.value(q)}")
print(f"uk = {[sol.value(u0), sol.value(u1), sol.value(u2)]}")
