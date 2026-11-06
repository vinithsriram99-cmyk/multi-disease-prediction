"""Friendly names for the input fields."""
KIDNEY = {"age": "Age (years)", "bp": "Blood pressure (mm Hg)", "sg": "Urine specific gravity", "al": "Albumin in urine (0-5)",
          "su": "Sugar in urine (0-5)", "rbc": "Red blood cells in urine", "pc": "Pus cells in urine", "pcc": "Pus cell clumps",
          "ba": "Bacteria in urine", "bgr": "Random blood glucose (mg/dL)", "bu": "Blood urea (mg/dL)",
          "sc": "Serum creatinine (mg/dL)", "sod": "Sodium (mEq/L)", "pot": "Potassium (mEq/L)", "hemo": "Hemoglobin (g/dL)",
          "pcv": "Packed cell volume", "wc": "White blood cell count (cells/cumm)", "rc": "Red blood cell count (millions/cmm)",
          "htn": "Hypertension", "dm": "Diabetes mellitus", "cad": "Coronary artery disease", "appet": "Appetite",
          "pe": "Pedal edema (foot swelling)", "ane": "Anemia"}
LIVER = {"Age": "Age (years)", "Gender": "Gender", "Total_Bilirubin": "Total bilirubin (mg/dL)",
         "Direct_Bilirubin": "Direct bilirubin (mg/dL)", "Alkaline_Phosphotase": "Alkaline phosphatase (IU/L)",
         "Alamine_Aminotransferase": "ALT / SGPT (IU/L)", "Aspartate_Aminotransferase": "AST / SGOT (IU/L)",
         "Total_Protiens": "Total proteins (g/dL)", "Albumin": "Albumin (g/dL)", "Albumin_and_Globulin_Ratio": "Albumin/Globulin ratio"}
PARKINSONS_HELP = ("These are 22 measurements from a patient's voice recording (pitch variation 'jitter', amplitude "
                   "variation 'shimmer', noise ratios and non-linear measures). They come from a speech-analysis "
                   "tool, so for the demo use the profile buttons or values from the dataset.")
LABELS = {"kidney": KIDNEY, "liver": LIVER, "parkinsons": {}}
