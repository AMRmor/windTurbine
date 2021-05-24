import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
from reliability.Fitters import Fit_Everything
from reliability.Distributions import Weibull_Distribution
from reliability.Other_functions import make_right_censored_data
from reliability.Distributions import Lognormal_Distribution, Gamma_Distribution, Weibull_Distribution, Mixture_Model
import matplotlib.pyplot as plt
from reliability.Fitters import Fit_Weibull_Mixture, Fit_Weibull_2P
from reliability.Other_functions import histogram, make_right_censored_data
import numpy as np
import matplotlib.pyplot as plt
degToRad = np.pi/180


headers = ['timestamp', 'Fruges Wind Speed [100 m]',
           'Fruges Wind Direction [100 m]']
df = pd.read_csv('Fruges.csv')

df['timestamp'] = df['timestamp'].map(
    lambda x: datetime.strptime(str(x), '%Y%m%dT%H%M'))
x = df['timestamp']
y = df['Fruges Wind Speed [100 m]']
# plot
fig2 = plt.plot(x, y)
fig2 = plt.show()
# CORRECTION DE LA VITESSE DU VENT EN FONCTION DE LA HAUTEUR 100m(mesure) à 84m(eolienne)
WSC = df['Fruges Wind Speed [100 m]']*((np.log(84/0.05))/np.log(100/0.05))
df['Fruges Wind Speed [100 m]'] = WSC
del df['Fruges Wind Direction [100 m]']
print(df)
plt.hist(df['Fruges Wind Speed [100 m]'], bins=50,
         color='purple', edgecolor='black')
plt.xlabel('vitesses du vent [m/s]')
plt.ylabel('répétition')
plt.title('Distribution des vitesses du vent à Fruges en 2019')
Vmoy = (np.mean(WSC))  # vitesse moyenne
print('La vitesse moyenne est ', Vmoy)
# ******************************************************************************


raw_data = np.asarray(WSC)
data = make_right_censored_data(raw_data, threshold=25)
results = Fit_Everything(failures=data.failures,
                         right_censored=data.right_censored)
# fit all the models
print('The best fitting distribution was', results.best_distribution_name,
      'which had parameters', results.best_distribution.parameters)

# create the mixture model
d1 = Lognormal_Distribution(mu=1.77932, sigma=0.524176)
d2 = Weibull_Distribution(alpha=7.48025, beta=2.32338, gamma=0.0361584)
d3 = Gamma_Distribution(alpha=1.50549, beta=4.42565, gamma=0.0361584)
mixture_model = Mixture_Model(
    distributions=[d1, d2, d3], proportions=[0.1, 0.8, 0.1])
# plot the 5 functions using the plot() function
mixture_model.plot()
# plot the PDF and CDF
plot_components = True  # this plots the component distributions. Default is False
plt.figure(figsize=(9, 5))
plt.subplot(121)
mixture_model.PDF(plot_components=plot_components, color='red', linestyle='--')
plt.subplot(122)
mixture_model.CDF(plot_components=plot_components, color='red', linestyle='--')
plt.subplots_adjust(left=0.1, right=0.95)
plt.show()


all_data = np.asarray(WSC)
data = make_right_censored_data(all_data, threshold=30)
# fit the Weibull Mixture and Weibull_2P
mixture = Fit_Weibull_Mixture(failures=data.failures, right_censored=data.right_censored,
                              show_probability_plot=False, print_results=False)
single = Fit_Weibull_2P(failures=data.failures, right_censored=data.right_censored,
                        show_probability_plot=False, print_results=False)
print('Weibull_Mixture BIC:', mixture.BIC, '\nWeibull_2P BIC:',
      single.BIC)  # print the goodness of fit measure
# plot the Mixture and Weibull_2P
histogram(all_data, white_above=30)
mixture.distribution.PDF(label='Weibull Mixture')
single.distribution.PDF(label='Weibull_2P')
plt.title('Comparison of Weibull_2P with Weibull Mixture')
plt.legend()
plt.show()

# METHODE FOURNI PAR MONSIEUR THIBAUT MENARD
data = np.loadtxt('NACA0018.txt', skiprows=12)
angle = [col[0]*degToRad for col in data]
cz = [col[1] for col in data]

cx = [col[2] for col in data]

cxf = interp1d(angle, cx, bounds_error=False)
print(cxf)
print(len(angle))
czf = interp1d(angle, cz, bounds_error=False)

Vz = 15
rmin = 0.3
rmax = 1.8
Lc = 0.2
rho = 1.2
Npales = 3
avd = 30*degToRad
avf = 30*degToRad
A = 15


def va(r, ap):
    return np.sqrt(Vz**2+(r*ap)**2)


def beta(r, ap):
    return np.arcsin(Vz/va(r, ap))


def i(r, ap):
    return beta(r, ap)-av(r)


def av(r):
    return avd+(r-rmin)/(rmax-rmin)*(avf-avd)


def f(r, ap):
    anglei = i(r, ap)
    print("-----------------------------------------------------")
    print(len(anglei))
    pb = beta(r, ap)
    term1 = -cxf(anglei)*np.cos(pb)
    term2 = czf(anglei)*np.sin(pb)
    term3 = 0.5*rho*va(r, ap)**2*Lc
    return (term1+term2)*term3


def Caero(ap):
    r = np.linspace(rmin, rmax, 20)
    y = r*f(r, ap)
    It = np.trapz(y, r)
    return It


def g(ap):
    return Npales*Caero(ap)-A*np.abs(ap)


plt.figure("g")
ap = np.linspace(1, 15)
yg = np.array([g(api) for api in ap])
fig1 = plt.plot(ap, yg)

vitrot = fsolve(g, 2)
print('ap=', ap)
print(vitrot)


# CALCUL A ET A'


r = np.linspace(rmin, rmax, 20)


def sigma(r):
    return Npales*Lc/2*np.pi*r
# C1=czf(anglei)*np.sin(pb)-cxf(anglei)*np.cos(pb)


def C1(r, ap):
    anglei = i(r, ap)
    pb = beta(r, ap)
    term1 = -cxf(anglei)*np.cos(pb)
    term2 = czf(anglei)*np.sin(pb)
    return (term1+term2)

# C2=czf(anglei)*np.cos(pb)+cxf(anglei)*np.sin(pb)


def C2(r, ap):
    anglei = i(r, ap)
    pb = beta(r, ap)
    term1 = cxf(anglei)*np.sin(pb)
    term2 = czf(anglei)*np.cos(pb)
    return (term1+term2)


def A1(r, ap):
    # anglei=i(r,ap)
    pb = beta(r, ap)
    term1 = (2*np.sin(pb))**2
    term2 = sigma(r)*C2(r, ap)
    return 1/((term1/term2)+1)


def A2(r, ap):
    # anglei=i(r,ap)
    pb = beta(r, ap)
    return 1/(((4*np.sin(pb)*np.cos(pb)*(r, ap))/sigma(r)*C1(r, ap))-1)


#print('a=', A1, 'et a1=', A2)
