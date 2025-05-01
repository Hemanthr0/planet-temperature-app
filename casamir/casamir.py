import numpy as np
import matplotlib.pyplot as plt

# Constants
speed_of_sound = 343  # speed of sound in air (m/s)
source_frequency = 1000  # frequency of the sound emitted by the source (Hz)

# Doppler formula for sound
def doppler_sound(v_source, v_observer, frequency):
    return frequency * (speed_of_sound + v_observer) / (speed_of_sound - v_source)

# Relative speeds (positive for moving towards, negative for moving away)
velocities = np.linspace(-100, 100, 100)  # observer moving towards and away from the source
frequencies = doppler_sound(0, velocities, source_frequency)  # assuming the source is stationary

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(velocities, frequencies, label="Observed Frequency")
plt.xlabel("Observer Velocity (m/s)")
plt.ylabel("Observed Frequency (Hz)")
plt.title("Doppler Effect for Sound")
plt.grid(True)
plt.legend()
plt.show()
def doppler_light(v, frequency, speed_of_light=3e8):
    return frequency * np.sqrt((1 + v / speed_of_light) / (1 - v / speed_of_light))

# Define parameters
light_source_frequency = 5e14  # frequency of visible light (Hz)

# Calculate Doppler shift for different velocities
velocities_light = np.linspace(-1e6, 1e6, 100)  # observer moving at high speeds (m/s)
frequencies_light = doppler_light(velocities_light, light_source_frequency)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(velocities_light, frequencies_light, label="Observed Frequency (Light)")
plt.xlabel("Observer Velocity (m/s)")
plt.ylabel("Observed Frequency (Hz)")
plt.title("Doppler Effect for Light")
plt.grid(True)
plt.legend()
plt.show()
