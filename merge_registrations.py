import pandas as pd


def merge_registrations(excel_file="paid_registrations_Nov3rd.xlsx", csv_file="Ablation_workshop_internal.csv"):

    # --- Read both files ---
    df_excel = pd.read_excel(excel_file)
    df_csv = pd.read_csv(csv_file)

    # --- Rename columns for consistency ---
    df_excel = df_excel.rename(columns={
        "MAE-FULL NAME": "Name",
        "MAE-INSTITUTIONNAME": "Institution"
    })
    df_csv = df_csv.rename(columns={
        "Full Name": "Name",
        "Affiliation": "Institution"
    })

    # --- Merge both dataframes ---
    # Option 1: concatenate (stack both vertically)
    merged_df = pd.concat([df_excel, df_csv], ignore_index=True)

    # --- Optional: drop duplicates based on Name and Institution ---
    merged_df = merged_df.drop_duplicates(subset=["Name", "Institution"])

    # --- Keep only relevant columns ---
    merged_df = merged_df[["Name", "Institution"]]

    # --- Display or save ---
    # print(merged_df.head())
    # merged_df.to_csv("merged_output.csv", index=False)
    return merged_df