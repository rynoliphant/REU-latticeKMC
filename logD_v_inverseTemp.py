import numpy as np
import matplotlib.pyplot as plt

D300 = np.array([5.7562, 5.8343, 5.7515]) * 10**(-10)
D500 = np.array([1.4119, 1.3591, 1.4267]) * 10**(-6)
D700 = np.array([3.7615, 3.7244, 3.7491]) * 10**(-5)
diff_average = np.array([np.mean(D300), np.mean(D500), np.mean(D700)])#* 10**(-4) #m^2/s
temp = np.array([300, 500, 700]) #K

m, b = np.polyfit(1/temp, np.log(diff_average), 1)
inverse_temp = 1/np.arange(290,710)

k_B = 8.617333 * 10**(-5) #eV/K
D_0 = np.exp(b)
D_out = D_0 * np.exp(m/(temp))
E_effective = -1*m*k_B
#print(diff_average)
#print(D_out)
print(E_effective)
print(D700)

fig = plt.figure(figsize=(32,24))
plt.plot(inverse_temp, m*inverse_temp + b, linewidth=8)
plt.plot(1/temp, np.log(diff_average), 'bo', markersize=30)
plt.xlabel('1/Temperature (K^-1)', size=60)
plt.xticks(fontsize=40)
plt.ylabel('Log(Diffusivity (m^2/s))', size=60)
plt.yticks(fontsize=50)
plt.title('Log(Diffusivity) vs 1/Temperature', size=70)
plt.text(1/400, np.log(np.mean(D700)),'Y = '+str(round(m,5))+'X + ('+str(round(b,5))+')', fontsize=40)
plt.savefig('../poster/logDiffusivity_v_inverseTemperature.png')
plt.show()