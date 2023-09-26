import os
import shutil
import pydicom

def find_dicom_files(directory):
    dicom_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    return dicom_files

def categorize_dicom_files(dicom_files):
    categorized_files = {"cine": [], "mag": [], "x": [], "y": [], "z": [], "unknown": []}
    
    for file in dicom_files:
        ds = pydicom.dcmread(file)
        if (0x0043, 0x1030) in ds:
            image_type = ds[(0x0043, 0x1030)].value
            if image_type == 0 or image_type == 7:
                categorized_files["cine"].append(file)
            elif image_type == 2 or image_type == 8:
                categorized_files["mag"].append(file)
            elif image_type == 3 or image_type == 9:
                categorized_files["x"].append(file)
            elif image_type == 4 or image_type == 10:
                categorized_files["y"].append(file)
            elif image_type == 5 or image_type == 11:
                categorized_files["z"].append(file)
            else:
                categorized_files["unknown"].append(file)
        else:
            categorized_files["unknown"].append(file)
            
    return categorized_files

                
def copy_categorized_files(categorized_files, output_directory):
    for category, files in categorized_files.items():
        category_directory = os.path.join(output_directory, category)
        os.makedirs(category_directory, exist_ok=True)
        for file in files:
            shutil.copy(file, category_directory)

if __name__ == "__main__":
    input_directory = ""
    output_directory = ""
    
    dicom_files = find_dicom_files(input_directory)
    categorized_files = categorize_dicom_files(dicom_files)
    copy_categorized_files(categorized_files, output_directory)
