import h5py
import ROOT
import numpy as np
import os

CH = 7
h5_output = "dataset_channel7_part_32500_32501.h5"

ROOT.gROOT.LoadMacro("DSB.C")

start_num = 32500
end_num = 32501
bad_events = [2137, 4467, 6832, 7073, 7319, 8242, 8524, 9200, 9424, 11066, 12134, 13892, 15019, 17990, 29539]

TreeReader = ROOT.TreeReader
tree = TreeReader("./outfile_LG.root")
#m = 32501
print(f"Processing events from {start_num} to {end_num}")

column_labels = [f"sample_{i}" for i in range(1024)] + ["T", "C", "S", "T_err", "C_err", "S_err"]

with h5py.File(h5_output, "w") as h5file:

    dataset = h5file.create_dataset(
        "Channel7", 
        shape=(0, 1024 + 6), 
        maxshape=(None, 1024 + 6), 
        dtype="float32"
    )
    dataset.attrs["column_titles"] = np.array(column_labels, dtype="S")


    for EVT in range(start_num, end_num):
        if EVT in bad_events:
            print(f"Skipping bad event: {EVT}")
            continue

        tree.get_entry(EVT)
        
        voltages = tree.voltages(CH)
        samples = np.array(voltages, dtype=np.float32)
        
        ROOT.reco_WF_SDL(EVT, CH)

        fFit = ROOT.gROOT.FindObject("fFit")
        T = fFit.GetParameter(0)  # Time jitter
        C = fFit.GetParameter(1)  # Nc x A1pe
        S = fFit.GetParameter(2)  # Ns x A1pe

        T_err = fFit.GetParError(0)  # Time jitter error
        C_err = fFit.GetParError(1)  # Nc x A1pe error
        S_err = fFit.GetParError(2)  # Ns x A1pe error

        ROOT.gROOT.GetListOfFunctions().Delete() 

        row = np.hstack([samples, [T, C, S, T_err, C_err, S_err]])

        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = row

        print(f"event: {EVT} is done.")

print(f"HDF5 file saved as {h5_output}.")
