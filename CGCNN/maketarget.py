import os
import csv
from pymatgen.io.cif import CifParser

# Path to the folder containing CIF files
folder_path = 'dataset/'
output_csv = 'qmof_energy_data.csv'

data = []

# Iterate over all CIF files
for filename in os.listdir(folder_path):
    if filename.endswith('.cif'):
        filepath = os.path.join(folder_path, filename)
        try:
            parser = CifParser(filepath)
            structure = parser.get_structures(primitive=False)[0]
            cif_dict = parser.as_dict()

            # Try extracting energy from CIF metadata
            energy = None
            for key in cif_dict:
                if "_dft_total_energy" in key or "energy" in key.lower():
                    energy = float(cif_dict[key])
                    break

            if energy is None:
                # Fallback: look for energy in data block
                for key, value in cif_dict.items():
                    if isinstance(value, str) and "energy" in value.lower():
                        try:
                            energy = float(value)
                            break
                        except ValueError:
                            continue

            # Store data
            qmof_name = filename.replace('.cif', '')
            data.append((qmof_name, energy))

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# Write to CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['qmof_name', 'energy_eV'])
    for row in data:
        writer.writerow(row)

print(f"CSV file saved as {output_csv}")
