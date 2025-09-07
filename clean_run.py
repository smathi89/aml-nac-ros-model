import pandas as pd
import os

def safe_read_csv(path, label):
    print(f"\n==== Loading dataset: {label} ====")
    try:
        df = pd.read_csv(path, encoding="utf-8")
        print(df)
        return df
    except Exception as e:
        print(f"ERROR loading {label} from {path}: {e}")
        return None

# Show what's in the /data folder
print("Files in /data folder:", os.listdir("data"))

# Load standard project datasets
df_cd14_ros = safe_read_csv("data/cd14_ros_foldchange_by_line.csv", "CD14 + ROS Fold Change by Cell Line")
df_combination_ci = safe_read_csv("data/combination_treatment_ci_summary.csv", "Combination Treatment CI Summary")
df_ic50_microscopy = safe_read_csv("data/compound_ic50_microscopy_summary.csv", "Compound IC50 & Microscopy Summary")
df_external_ros_nac = safe_read_csv("data/external_ros_nac_data.csv", "External ROS + NAC Data (Merged)")
df_iga_peak = safe_read_csv("data/peak_iga_response_by_group.csv", "Peak IgA Response by Group")
df_np_size = safe_read_csv("data/np_size_zeta.csv", "Nanoparticle Size & Zeta Potential")
df_patient_cd34 = safe_read_csv("data/patient_cd34_variant_summary.csv", "Patient CD34 Variant Summary")
df_peitc_nac = safe_read_csv("data/peitc_nac_growth_response.csv", "PEITC + NAC Growth Response")
df_ros_158n = safe_read_csv("data/ros_nac_158n_vs_158jp.csv", "ROS NAC in 158N vs 158JP")
df_ros_cd14 = safe_read_csv("data/ros_and_cd14_aml_cell_lines.csv", "ROS + CD14 AML Cell Lines")
df_ros_rescue = safe_read_csv("data/ros_nac_rescue_158n.csv", "ROS NAC Rescue in 158N")
df_ros_timecourse = safe_read_csv("data/ros_timecourse_as2o3_emodin_nac.csv", "ROS Timecourse: As2O3, Emodin, NAC")
df_sas_diameters = safe_read_csv("data/sas_diameter_measurements.csv", "S/AS Diameter Measurements")
df_sample_props = safe_read_csv("data/sample_distribution_properties.csv", "Sample Distribution Properties")
df_sphingolipid = safe_read_csv("data/sphingolipid_thp1_nomo1.csv", "Sphingolipid THP1")
df_transfection_zeta = safe_read_csv("data/transfection_agent_zeta.csv", "Transfection Agent Zeta Potential")
df_uptake_f8 = safe_read_csv("data/uptake_timecourse_f8_f14.csv", "Uptake Timecourse: F8 to F14")
df_uptake_pn = safe_read_csv("data/uptake_timecourse_pn_variants.csv", "Uptake Timecourse: PN Variants")

# Load new merged datasets (5 subtopics)
df_merged_conditions = safe_read_csv("data/merged_specific_cell_conditions.csv", "Merged: Specific Cell Lines + Conditions")
df_external_nac = safe_read_csv("data/external_ros_nac_data.csv", "Merged: External ROS + NAC Levels")
df_peg_nac = safe_read_csv("data/peg_nac_nanoparticle_dataset.csv", "Merged: PEG-NAC Nanoparticles")
df_nac_cancer = safe_read_csv("data/nac_ros_cancer_conditions_dataset.csv", "Merged: NAC + ROS Cancer Conditions")
df_aml_final = safe_read_csv("data/AML_CellLine_NAC_ROS_Data_with_PEGylated.csv", "Merged: AML CellLine NAC ROS + PEGylated")

# Final debug block for ROS+ cells by drug siRNA
ros_by_drug_path = "data/ros_positive_cells_by_drug_siRNA.csv"
print(f"\nChecking: {ros_by_drug_path}")

if os.path.exists(ros_by_drug_path):
    print("File exists! Showing raw content:\n")
    try:
        with open(ros_by_drug_path, "r", encoding="utf-8") as f:
            print(f.read())
    except Exception as read_error:
        print(f"Could not read file: {read_error}")

    df_ros_by_drug_sirna = safe_read_csv(ros_by_drug_path, "ROS+ Cells by Drug and siRNA")
else:
    print("File not found:", ros_by_drug_path)

print("\n==== DONE ====")







