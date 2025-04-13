# playing around with filters, fft and images.
# 
# i can also start implementing histogram processing stuff here

# TODO: 
# 1. Can we design a butterworth filter?

import numpy
import numpy.fft as fft
import matplotlib.pyplot as plt

f = numpy.zeros((1000, 1000))

f[0:101,0:101] = 1
f[899:1000,899:1000] = 1

print(f)

# get f's 2D Fourier transform
F = fft.fft2(f)
#print(F)

Fs = fft.fftshift(F)
#print(Fs)

print('F[0,0] = ', F[0,0])
print('Fs[500,500] = ', Fs[500,500])
# construct ideal low-pass filter
H = numpy.zeros((1000, 1000))

dc_index = 500
span = 11
H[dc_index - span:dc_index + span+1, dc_index - span:dc_index + span+1] = 1

# construct an extra high-pass filter from the filter we already have
Hh = 1 - H

# apply low-pass
Gs = H * Fs

# get g, initial image but with a low-pass filter applied
G = fft.ifftshift(Gs)

g = fft.ifft2(G)

print('g = ', g)
g = numpy.astype(g, float)

# get i, initial image but with a high-pass filter applied
Is = Hh * Fs

I = fft.ifftshift(Is)

i = fft.ifft2(I)

print('i = ', i)
i = numpy.astype(i, float)

# plot everything
plt.figure(0)
plt.pcolormesh(f)


plt.figure(1)
plt.pcolormesh(g)

plt.figure(2)
plt.pcolormesh(i)
#plt.show()
