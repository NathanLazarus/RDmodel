from casadi import *
import numpy as np
from csv import reader
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
warnings.filterwarnings("ignore")

M = 5
alpha = 0.32
delta = 0.08
beta = 0.98
G = 1.014
eta = 2
H = 31.1034083778894
zstar = 1
deltaRD = 0.25
# P_f = DM.ones(M)*(1-1/np.array(exp(range(M))))
P_s = np.array([0.3,0.2,0.1,0.05,0.01])
h = 0
P_f = DM.ones(M)*(1-P_s)+P_s*h
print(P_f)
# P_f1 = 0.9
# P_f2 = 0.9
chi = 4.3358
xi = 20
basicGoodProductivity = 1 #5.0619*0.0899541942

vars = [
    "cBs",
    "cTs",
    "lBs",
    "ls",
    "kBs",
    "ks",
    "finalprice"
]

vectorvars = [
    "lT",
    "kT",
    "intermediateprice",
    "Psi",
    "rdx",
    "rdStock" #,
    # "totalprofits"
]

for var in vars:
    globals()[var] = SX.sym(var)

for vectorvar in vectorvars:
    globals()[vectorvar] = SX.sym(vectorvar, M, 1)

# cBs = SX.sym("cBs")
# cTs = SX.sym("cTs")
# lBs = SX.sym("lBs")
# lT1 = SX.sym("lT1")
# lT2 = SX.sym("lT2")
# ls = SX.sym("ls")
# kBs = SX.sym("kBs")
# kT1 = SX.sym("kT1")
# kT2 = SX.sym("kT2")
# ks = SX.sym("ks")
# intermediateprice1 = SX.sym("intermediateprice1")
# intermediateprice2 = SX.sym("intermediateprice2")
# finalprice = SX.sym("finalprice")
# rdx = SX.sym("rdx")
# rdStock = SX.sym("rdStock")
# Psi  = SX.sym("Psi")
# rdxB = SX.sym("rdxB")
# rdxT = SX.sym("rdxT")

x = vertcat(*[globals()[var] for var in vars],*[globals()[vectorvar] for vectorvar in vectorvars])

x0 = DM.ones(x.shape[0])
x0[14] = 1 #1e4
print(x0)
obj = -cBs - cTs

# Psi1 = 1
# Psi2 = 1
# rdx1 = 0
# rdx2 = 0
# theta1 = chi*(1 - 1/M)
# theta2 = chi*(1 - 1/M)
# theta1 = 3
# theta2 = 10000
# theta = DM.ones(M)*np.array(exp(range(M))+1)
theta = DM.ones(M)*12
r = alpha * basicGoodProductivity * zstar * kBs ** (alpha-1) * lBs ** (1 - alpha)
w = (1 - alpha) * basicGoodProductivity * zstar * kBs ** alpha * lBs ** (-alpha)
yB = basicGoodProductivity * zstar * kBs ** alpha * lBs ** (1 - alpha)
# wT1 = intermediateprice1 * ((theta1 - 1) / theta1) * (1 - alpha) * Psi1 * zstar * kT1 ** alpha * lT1 ** (-alpha)
# rT1 = intermediateprice1 * ((theta1 - 1) / theta1) * alpha * Psi1 * zstar * kT1 ** (alpha-1) * lT1 ** (1-alpha)
# yT1 = Psi1 * zstar * (kT1) ** alpha * (lT1) ** (1 - alpha)
# pi1 = intermediateprice1 * yT1 - (wT1 * lT1 + rT1 * kT1 + rdx1)
wT = intermediateprice * ((theta - 1) / theta) * (1 - alpha) * Psi * zstar * kT ** alpha * lT ** (-alpha)
rT = intermediateprice * ((theta - 1) / theta) * alpha * Psi * zstar * kT ** (alpha-1) * lT ** (1-alpha)
yT = Psi * zstar * (kT) ** alpha * (lT) ** (1 - alpha)
pi = intermediateprice * yT - (wT * lT + rT * kT + rdx)

householdbudget = ((G - 1 + delta) * ks + cBs + finalprice * cTs) - (w * ls + r * ks + sum1(pi))
intertempEuler = 1 - (beta / G) * (1 + r - delta)
acrossGoodEuler = cBs - finalprice * cTs / xi
labMkt = H * ls ** eta * cBs - w
kidentity = ks - (sum1(kT) + kBs)
lidentity = ls - (sum1(lT) + lBs)
basicGoodExpenditure = yB - ((G - 1 + delta) * ks + cBs + sum1(rdx))
technologicalGoodExpenditure = sum1(intermediateprice * yT) - (finalprice * cTs)
factorPricesW = wT - w
factorPricesR = rT - r
pricingAggregation = finalprice - sum1(intermediateprice**(1-chi)) ** (1/(1-chi))
demandfunc = yT - intermediateprice ** (-chi) * (sum1(intermediateprice**(1-chi))) ** (chi/(1-chi)) * cTs
rdtransition = G * rdStock - ((1-deltaRD * P_f)*rdStock + rdx)
rdtoPsi = Psi - rdStock*10
optimalRD = (1/(1+r))*(theta - 1)/(theta)*(1/Psi)*intermediateprice*yT - (1 - (1/(1+r))*(1-deltaRD*P_f))
# totalprofitsSum = totalprofits - pi
# rdAggregation = (rdxB + rdxT) - rdx

constraint = vertcat(
    householdbudget,
    intertempEuler,
    acrossGoodEuler,
    labMkt,
    kidentity,
    lidentity,
    basicGoodExpenditure,
    technologicalGoodExpenditure,
    factorPricesW,
    factorPricesR,
    pricingAggregation,
    demandfunc,
    rdtransition,
    rdtoPsi,
    optimalRD #,
    # totalprofitsSum
)


nlp = {
    "x": x,
    "f": obj,
    "g": constraint,
}
solver = nlpsol("solver", "ipopt", nlp, {"ipopt.print_level": 0, "ipopt.tol": 1e-10})
with suppress_stdout():
    solution = solver(x0=x0, lbg=-1e-13, ubg=1e-13, lbx = 0, ubx = 1e12)

numberedVarnames = vars+[vectorvar+str(ind) for vectorvar in vectorvars for ind in range(1,M+1)]
print(dict(zip(
    numberedVarnames,
    np.array(solution["x"]).flatten().tolist()
)))
# print(solution['g'])
print('total constraint violation = ',sum1(solution['g']))
print(solution['x'])

for i in range(len(numberedVarnames)):
    globals()[numberedVarnames[i]] = np.array(solution["x"]).flatten().tolist()[i]

yT1 = Psi1 * zstar * (kT1) ** alpha * (lT1) ** (1 - alpha)
yT2 = Psi2 * zstar * (kT2) ** alpha * (lT2) ** (1 - alpha)
yT3 = Psi3 * zstar * (kT3) ** alpha * (lT3) ** (1 - alpha)
yT4 = Psi4 * zstar * (kT4) ** alpha * (lT4) ** (1 - alpha)
r = alpha * basicGoodProductivity * zstar * kBs ** (alpha-1) * lBs ** (1 - alpha)
w = (1 - alpha) * basicGoodProductivity * zstar * kBs ** alpha * lBs ** (-alpha)
pi1 = intermediateprice1 * yT1 - (w * lT1 + r * kT1 + rdx1)
pi2 = intermediateprice2 * yT2 - (w * lT2 + r * kT2 + rdx2)
pi3 = intermediateprice3 * yT3 - (w * lT3 + r * kT3 + rdx3)
pi4 = intermediateprice4 * yT4 - (w * lT4 + r * kT4 + rdx4)
print(pi1,pi2,pi3,pi4)
# yT4 = Psi4 * zstar * (kT4) ** alpha * (lT4) ** (1 - alpha)
# yT5 = Psi5 * zstar * (kT5) ** alpha * (lT5) ** (1 - alpha)

# print(intermediateprice1*yT1,intermediateprice2*yT2,intermediateprice3*yT3) #,intermediateprice4*yT4,intermediateprice5*yT5)
print(intermediateprice2)
# print(theta1)
# print(1/theta1)
# print(yT*M*intermediateprice)
# print(cTs*finalprice)
# print(solution['g'])
# print(dict(zip(
#   ["householdbudget","intertempEuler","acrossGoodEuler","labMkt",
#   "kidentity","lidentity","basicGoodExpenditure","technologicalGoodExpenditure",
#   "factorPricesW","factorPricesR","rdtransition","rdtoPsi","optimalRD"],
#   np.array(solution["g"]).flatten().tolist())))