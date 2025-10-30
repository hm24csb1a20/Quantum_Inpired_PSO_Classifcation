import zipfile
import os
import shutil
from final_all_combined import run_pipeline


if (__name__=="__main__"):
        
    # extracting for dataset 1 
    main_zip_path = "archive.zip"
    extract_root = "extracted_dataset"
    final_dataset_path = "dataset1"
    with zipfile.ZipFile(main_zip_path, 'r') as f: f.extractall(extract_root)

    for r, d, _ in os.walk(extract_root):
        if os.path.basename(r).lower() == "images":
            if os.path.exists(final_dataset_path): shutil.rmtree(final_dataset_path)
            shutil.copytree(r, final_dataset_path)
            break

    print("dataset1 is ready in:", os.path.abspath(final_dataset_path))


    # extacitng for dataset 2 
    main_zip_path = "IIITDMJ_Smoke.zip"
    extract_root = "extracted_dataset"
    final_dataset_path = "dataset2"
    with zipfile.ZipFile(main_zip_path, 'r') as f: f.extractall(extract_root)

    for r, d, _ in os.walk(extract_root):
        if os.path.basename(r).lower() == "train":
            if os.path.exists(final_dataset_path): shutil.rmtree(final_dataset_path)
            shutil.copytree(r, final_dataset_path)
            break

    print("dataset2 is ready in:", os.path.abspath(final_dataset_path))


    #extracting the dataset 3  into the local scope 
    main_zip_path = "GastroEndoNet Comprehensive Endoscopy Image Dataset for GERD and Polyp Detection.zip"
    extract_root = "extracted_dataset"
    final_dataset_path = "dataset3"

    with zipfile.ZipFile(main_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_root)

    for r, d, files in os.walk(extract_root):
        for file in files:
            if file.lower().endswith("original image.zip"):
                original_zip_path = os.path.join(r, file)
                

            
                with zipfile.ZipFile(original_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(final_dataset_path)
                
                break
        else:
            continue
        break

    print("dataset3 is ready in:", os.path.abspath(final_dataset_path))

    # finally running the QIGPSO image classifcation 


    # for dataset no 1 should get 92.80
    run_pipeline(
        DATA_ROOT_DIR="dataset1",
        MODEL_CACHE_FILENAME="dataset1",
        FEATURE_CACHE_NAME="dataset1.npz"
    )

    # for dataset no 2 shold get 90.60
    run_pipeline(
        DATA_ROOT_DIR="dataset2",
        MODEL_CACHE_FILENAME="dataset2",
        FEATURE_CACHE_NAME="dataset2.npz"
    )

    # for dataset no 3 should get 72.80
    run_pipeline(
        DATA_ROOT_DIR="dataset3",
        MODEL_CACHE_FILENAME="dataset3",
        FEATURE_CACHE_NAME="dataset3.npz"
    )
