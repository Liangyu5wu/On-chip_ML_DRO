import h5py
import ROOT
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
from scipy.stats import norm

CH = 3
#m = 32501
m = 32501
guessing_num  = 80

skip_events = {2137, 4467, 6832, 7073, 7319, 8242, 8524, 9200, 9424, 11066, 12134, 13892, 15019, 17990, 29539}

ROOT.gROOT.LoadMacro("DSB.C")


TreeReader = ROOT.TreeReader
tree = TreeReader("./outfile_LG.root")

event_ids = []
mean_values = []
voltage_at_guess = []
diff_values = []
strange_ids = []
strange_v_ids = []


ratio_points_below_list = []



for EVT in range(0, m):
    if EVT in skip_events:
        continue

    tree.get_entry(EVT)

    voltages = np.array(tree.voltages(CH)) 
    times = np.array(tree.time())
    ROOT.gROOT.GetListOfFunctions().Delete()

    mask = (times >= 0) & (times <= 50)
    selected_voltages = voltages[mask]

    mean_val = np.mean(selected_voltages)

    closest_idx = np.argmin(np.abs(times - guessing_num))
    voltage_closest = voltages[closest_idx]

    diff_value = voltage_closest - mean_val

    count_below_num = np.sum(times < 90)
    total_count = len(times)
    ratio_count_below_num = count_below_num/total_count

    ratio_points_below_list.append(ratio_count_below_num)

    if (np.abs(diff_value)>10):
        strange_ids.append(EVT)
    
    if (mean_val < 300):
        strange_v_ids.append(EVT)

    event_ids.append(EVT)
    mean_values.append(mean_val)
    voltage_at_guess.append(voltage_closest)
    diff_values.append(diff_value)

    if EVT % 1000 == 0:
        print(f"Processed event: {EVT}")

event_ids = np.array(event_ids)
mean_values = np.array(mean_values)
voltage_at_guess = np.array(voltage_at_guess)
diff_values = np.array(diff_values)


ratio_points_below_list = np.array(ratio_points_below_list)


print(strange_ids)
print(strange_v_ids)
print(ratio_points_below_list)





# plt.figure(figsize=(10, 6))
# plt.scatter(event_ids, mean_values, marker='o',  label="Mean Voltage (0-50 ns)", color='blue', s=8)
# plt.scatter(event_ids, voltage_at_guess, color='red', marker='x', label="Voltage at Guessing_num", s=8)
# plt.scatter(event_ids, diff_values,marker='o',  label="Voltage Difference", color='green', s=8)

# plt.xlabel("Event ID")
# plt.ylabel("Voltage (mV)")
# plt.title("Voltage Analysis over Events")
# plt.legend()
# plt.grid(True)
# plt.xlim(-10, m + 50) 
# plt.show()


mu, sigma = np.mean(ratio_points_below_list), np.std(ratio_points_below_list)
bins=80
    

counts, bin_edges = np.histogram(ratio_points_below_list, bins=bins, density=True)

plt.figure(figsize=(8, 6))
plt.hist(ratio_points_below_list, bins=bins,  alpha=0.6, color='b', label='Histogram')
    
plt.xlabel('Ratio of counts before 80 ns')
plt.ylabel('Counts')
plt.legend()
plt.grid()
plt.show()
