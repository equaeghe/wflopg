"""Script to generate WES-paper wake loss percentages for Horns Rev."""

# wake loss percentages for WES paper optimizations
wlp_WES = {'original': 8.4146, 'hexagonal': 8.3136}

# wakeless expected power values for the sites
# (1.0826803 (expected wakeless power) * number of turbines)
power_wakeless = 87.18902454  # TODO: Feng & Shen used 88.17604114709 (Table 4)

# Power values from Feng & Shen 2015 paper
power_orginal = 81.6720  # TODO: I find 79.71865733!
wlp_original = 100 * (1 - power_orginal/power_wakeless)
print(wlp_original)

power_optimized = {'original': 81.8927, 'hexagonal': 81.9770}
wlp_optimized = {}
for key in power_optimized.keys():
    wlp_optimized[key] = 100 * (1 - power_optimized[key]/power_wakeless)
print(wlp_optimized)
