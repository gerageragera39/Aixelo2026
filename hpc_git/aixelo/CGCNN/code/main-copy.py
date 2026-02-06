from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os, subprocess, shutil, json, re
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from fastapi.responses import StreamingResponse
import subprocess
import csv

UPLOAD_DIR = r"F:\Aixelo\Docker_Networks\uploads"
PB4_EXE = r"F:\Aixelo\Docker_Networks\poreblazer_gfortran.exe"
DEFAULTS_DAT = r"F:\Aixelo\Docker_Networks\defaults.dat"
UFF_ATOMS = r"F:\Aixelo\Docker_Networks\UFF.atoms"

# === MongoDB connection ===
client = MongoClient("mongodb://localhost:27017/")  # local MongoDB
db = client["mof_dashboard"]  # database
collection = db["cif_results"]  # collection

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

# Serve JSON files and other uploads
# Serve frontend

# Serve your uploads JSON
# app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# Serve your frontend (index.html, script.js, style.css)
# app.mount("/frontend", StaticFiles(directory="static", html=True), name="frontend")

def convert_cif_to_xyz(cif_path, xyz_path):
    subprocess.run(["obabel", cif_path, "-O", xyz_path], check=True)
    return xyz_path


def create_input_dat(cif_path, xyz_path, input_path):
    a = b = c = alpha = beta = gamma = None
    with open(cif_path, "r") as f:
        for line in f:
            if "_cell_length_a" in line: a = line.split()[1]
            if "_cell_length_b" in line: b = line.split()[1]
            if "_cell_length_c" in line: c = line.split()[1]
            if "_cell_angle_alpha" in line: alpha = line.split()[1]
            if "_cell_angle_beta" in line: beta = line.split()[1]
            if "_cell_angle_gamma" in line: gamma = line.split()[1]

    if None in [a, b, c, alpha, beta, gamma]:
        raise ValueError("Missing cell info in CIF")

    with open(input_path, "w") as f:
        f.write(f"{xyz_path}\n")
        f.write(f"{a}   {b}   {c}\n")
        f.write(f"{alpha}   {beta}   {gamma}\n")
    return input_path


def run_poreblazer(input_path, working_dir):
    shutil.copy(DEFAULTS_DAT, working_dir)
    shutil.copy(UFF_ATOMS, working_dir)

    subprocess.run([PB4_EXE, os.path.basename(input_path)], cwd=working_dir, check=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    possible_files = [
        os.path.join(working_dir, f"{base_name}_summary.dat"),
        os.path.join(working_dir, "summary.dat"),
        os.path.join(working_dir, "outputs", f"{base_name}_summary.dat"),
        os.path.join(working_dir, "outputs", "summary.dat")
    ]

    for file in possible_files:
        if os.path.exists(file):
            return file

    raise FileNotFoundError("PoreBlazer did not generate a summary file")


def summary_dat_to_json(summary_path, json_path):
    mapping = {
        "C": "XYZ file",
        "V_A^3": "System volume (Å³)",
        "M _g/mol": "System mass (g/mol)",
        "RHO_g/cm^3": "System density (g/cm³)",
        "PLD_A": "Pore limiting diameter (Å)",
        "LCD_A": "Maximum pore diameter (Å)",
        "D": "Dimension",
        "S_AC_A^2": "Total surface area (Å²)",
        "S_AC_m^2/cm^3": "Surface area per volume (m²/cm³)",
        "S_AC_m^2/g": "Surface area per mass (m²/g)",
        "V_He_A^3": "Helium volume (Å³)",
        "V_He_cm^3/g": "Helium volume per mass (cm³/g)",
        "V_G_A^3": "Geometric volume (Å³)",
        "V_G cm^3/g": "Geometric volume per mass (cm³/g)",
        "V_PO A^3": "Probe-occupiable volume (Å³)",
        "V_PO cm^3/g": "Probe-occupiable volume per mass (cm³/g)",
        "FV_PO": "Fraction of free volume"
    }

    ignore_lines = {"Total", "Network-accessible"}

    data = {}
    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line in ignore_lines:
                continue

            for short, full_name in mapping.items():
                if line.startswith(short):
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    if nums:
                        data[full_name] = nums[-1]

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, indent=4, ensure_ascii=False)

    return data

@app.post("/upload_cif/")
async def upload_cif(file: UploadFile = File(...)):
    try:
        cif_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(cif_path, "wb") as f:
            f.write(await file.read())

        base_name = os.path.splitext(file.filename)[0]
        xyz_path = os.path.join(UPLOAD_DIR, f"{base_name}.xyz")
        input_path = os.path.join(UPLOAD_DIR, f"{base_name}_input.dat")

        convert_cif_to_xyz(cif_path, xyz_path)
        create_input_dat(cif_path, xyz_path, input_path)
        summary_file = run_poreblazer(input_path, UPLOAD_DIR)

        json_path = os.path.join(UPLOAD_DIR, f"{base_name}_summary.json")
        summary_data = summary_dat_to_json(summary_file, json_path)

        # === Save result into MongoDB ===
        doc = {
            "material": base_name,
            "cif": file.filename,
            "summary": summary_data,
        }
        collection.insert_one(doc)

        return JSONResponse({
            "material": base_name,
            "cif": file.filename,
            "summary": summary_data,
            "json_file": os.path.basename(json_path),
            "json_url": f"/static/{os.path.basename(json_path)}"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
CGCNN_DIR = r"F:\Aixelo\Docker_Networks\CGCNN"
PREDICT_PY = os.path.join(CGCNN_DIR, "predict.py")

@app.post("/predict/")
async def run_prediction():
    def run_and_stream():
        process = subprocess.Popen(
            ["python", "predict.py"],
            cwd="F:\Aixelo\Docker_Networks\CGCNN",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in iter(process.stdout.readline, ''):
            yield line
        process.wait()
        yield "\n✅ Prediction finished!\n"

    # Stream logs AND after finishing, also serve predictions
    return StreamingResponse(run_and_stream(), media_type="text/plain")

@app.get("/predictions_data/")
async def get_predictions():
    file_path = r"F:\Aixelo\Docker_Networks\CGCNN\output.csv"
    data = []
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 3:  # make sure we have all 3 columns
                data.append({
                    "cif": row[0],                     # CIF name
                    "predicted": float(row[1]),        # Predicted value
                    "actual": float(row[2])            # Actual value
                })
    return data
@app.post("/predict_new_cif/")
async def predict_new_cif(file: UploadFile = File(...), actual: float = None):
    try:
        # Save the new CIF
        cif_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(cif_path, "wb") as f:
            f.write(await file.read())

        base_name = os.path.splitext(file.filename)[0]

        # Check predictions CSV
        predictions_file = r"F:\Aixelo\Docker_Networks\CGCNN\results\predictions.csv"
        predicted_value = None
        with open(predictions_file, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) < 2:
                    continue  # skip empty or malformed rows
                name_in_csv = row[0].split(".")[0]  # remove extension if present
                if name_in_csv == base_name:
                    predicted_value = float(row[1])
                    break

        if predicted_value is None:
            return {"error": "Prediction not found for this CIF"}

        return {
            "cif": file.filename,
            "predicted": predicted_value,
            "actual": actual,
            "error": abs(predicted_value - actual) if actual is not None else None
        }

    except Exception as e:
        return {"error": str(e)}


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import torch
from model import CrystalGraphConvNet
from data import CIFData, collate_pool
from torch.utils.data import DataLoader

app = FastAPI()

# Path to your pretrained model
MODEL_PATH = r"F:\Aixelo\Docker_Networks\CGCNN\model_best.pth.tar"
CGCNN_DIR = r"F:\Aixelo\Docker_Networks\CGCNN"

# Load your model once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)
model_args = checkpoint['args']

# Create model architecture
dummy_struct, _, _ = CIFData(CGCNN_DIR)[0]
orig_atom_fea_len = dummy_struct[0].shape[-1]
nbr_fea_len = dummy_struct[1].shape[-1]

model = CrystalGraphConvNet(
    orig_atom_fea_len, nbr_fea_len,
    atom_fea_len=model_args.atom_fea_len,
    n_conv=model_args.n_conv,
    h_fea_len=model_args.h_fea_len,
    n_h=model_args.n_h,
    classification=(model_args.task == 'classification')
)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()


@app.post("/predict_new_cif/")
async def predict_new_cif(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = os.path.join(CGCNN_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load CIF file as dataset
        dataset = CIFData(CGCNN_DIR, disable_save_torch=True)
        collate_fn = collate_pool
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        # Run prediction (just the last uploaded CIF)
        predicted_value = None
        for i, (input_, target, batch_cif_ids) in enumerate(loader):
            if batch_cif_ids[0] != os.path.splitext(file.filename)[0]:
                continue
            with torch.no_grad():
                if torch.cuda.is_available():
                    input_var = (input_[0].to(device), input_[1].to(device),
                                 input_[2].to(device),
                                 [crys_idx.to(device) for crys_idx in input_[3]])
                else:
                    input_var = input_
                output = model(*input_var)
                # Assuming regression task
                predicted_value = output.item()
            break

        if predicted_value is None:
            return JSONResponse({"error": "Failed to get prediction"}, status_code=500)

        return {"predicted": predicted_value}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
