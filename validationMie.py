#%%
# latex rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# .svg plot rendering
%config InlineBackend.figure_format = 'svg'


import matplotlib.pyplot as plt
import numpy as np

# initial a figure

yPx = 4000
xPx = 3000
image = np.zeros((yPx, xPx))


# generate a gaussian distribution where th integral of the distribution is 1. Parametrize it so to control the width of the distribution and the position of the peak
def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

mu = 1500
sigma = 10
x = np.linspace(0, xPx -1, xPx)
y = gaussian(x, mu, sigma)

noOfColors = np.linspace(0, 1, 5)
colormap = plt.cm.get_cmap('viridis', len(noOfColors))
colors = colormap(noOfColors)

# plot the gaussian distribution
plt.figure(figsize=(3, 2))
plt.plot(x, y, color=colors[0], zorder=100)
plt.title('Gaussian distribution')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
# plt.xlim(0, xPx)
# plt.ylim(0, 1.05)
# minor grid    
plt.grid(which='major', linestyle=':', linewidth='0.5', color='gray',zorder=0)

sigmaStart = 100
sigmaEnd = 500
nCurve = 100
sigmaArray = np.linspace(sigmaStart, sigmaEnd, nCurve)
gaussianArray = np.zeros((len(sigmaArray), xPx))

for i, sigma in enumerate(sigmaArray):
    gaussianArray[i, :] = gaussian(x, mu, sigma)    
    # made each curve to has an integral of 1
    gaussianArray[i, :] = gaussianArray[i, :] / np.sum(gaussianArray[i, :])
# plot the gaussian distribution

colormap = plt.cm.get_cmap('viridis', nCurve)
noOfColors = np.linspace(0, 1, nCurve)
colors = colormap(noOfColors)

plt.figure(figsize=(3, 2))
for i, sigma in enumerate(sigmaArray):
    plt.plot(x, gaussianArray[i, :], color=colors[i], zorder=100,linewidth=0.5)
plt.title('Gaussian distribution')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
# plt.xlim(0, xPx)
# plt.ylim(0, 1.05)
# minor grid
plt.grid(which='major', linestyle=':', linewidth='0.5', color='gray',zorder=0)
#%%generete the image with the gaussian distribution

sigmaStart = 500
sigmaEnd = 100
mu = 1500
nCurve = yPx
sigmaArray = np.linspace(sigmaStart, sigmaEnd, nCurve)
gaussianArray = np.zeros((len(sigmaArray), xPx))

for i, sigma in enumerate(sigmaArray):
    gaussianArray[i, :] = gaussian(x, mu, sigma)    
    # made each curve to has an integral of 1
    gaussianArray[i, :] = gaussianArray[i, :] / np.sum(gaussianArray[i, :])
    image[i, :] = gaussianArray[i, :]
jetConc = image + 1
plt.figure(figsize=(3, 4))
plt.imshow(jet, cmap='viridis')
plt.title('Gaussian distribution')
plt.colorbar()
# plt.xlim(0, xPx)
jetConcNorm = (jetConc - np.min(jetConc)) / (np.max(jetConc) - np.min(jetConc))

#%%
sigmaStart = 300
sigmaEnd = 700
mu = 1500
nCurve = yPx
sigmaArray = np.linspace(sigmaStart, sigmaEnd, nCurve)
gaussianArray = np.zeros((len(sigmaArray), xPx))

for i, sigma in enumerate(sigmaArray):
    gaussianArray[i, :] = gaussian(x, mu, sigma)    
    # made each curve to has an integral of 1
    gaussianArray[i, :] = gaussianArray[i, :] / np.sum(gaussianArray[i, :])
    image[i, :] = gaussianArray[i, :]
light = image + 1
plt.figure(figsize=(3, 4))
plt.imshow(light, cmap='viridis')
plt.title('Light Gaussian distribution')
plt.colorbar()

smoke = light * 300


# plt.xlim(0, xPx)
jetDef = jetConc*light
smokeDef = smoke*light

def concCharline(jet, smoke, light):
    charline = jet + smoke + light
    normCharline = (charline - np.min(charline)) / (np.max(charline) - np.min(charline))
    
    return normCharline

def concCarlone(jet, light):
    carlo = (jet-light)/light
    normCarlo = (carlo - np.min(carlo)) / (np.max(carlo) - np.min(carlo))
    return normCarlo
height = 0
charline = concCharline(jetDef, smokeDef, light)
carlo = concCarlone(jetDef, light)
plt.figure()
plt.plot(jetConcNorm[height, :], label='jet GT')
plt.plot(charline[height, :], label='charline')
plt.plot(carlo[height, :], label='carlo', linestyle='--')

plt.legend()

#%%
plt.subplot(1, 2, 1)
plt.imshow(charline, cmap='viridis')
plt.title('Charline')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(carlo, cmap='viridis')
plt.title('Carlo')
plt.colorbar()


