import os
from pathlib import Path

def create_project_structure(root_dir):
    
    # Define the directory structure
    structure = {
        'README.md': None,
        'requirements.txt': None,
        '.gitignore': None,
        'data': {
            'Fraud_Data.csv': None,
            'creditcard.csv': None,
            'IpAddress_to_Country.csv': None,
            'README.md': None
        },
        'notebooks': {
            '01_EDA.ipynb': None,
            '02_Feature_Engineering.ipynb': None,
            '03_Modeling.ipynb': None,
            '04_Explainability.ipynb': None
        },
        'src': {
            'data_preprocessing.py': None,
            'feature_engineering.py': None,
            'model_training.py': None,
            'utils.py': None
        },
        'docs': {
            'report.md': None,
        },
        'experiments': {}
    }

    # Create the directories and files
    for path, content in structure.items():
        if isinstance(content, dict):
            # It's a directory
            dir_path = os.path.join(root_dir, path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Create files inside the directory
            for file_name, file_content in content.items():
                file_path = os.path.join(dir_path, file_name)
                with open(file_path, 'w') as f:
                    if file_content is not None:
                        f.write(file_content)
        else:
            # It's a file   
            file_path = os.path.join(root_dir, path)
            with open(file_path, 'w') as f:
                if content is not None:
                    f.write(content)


if __name__ == "__main__":
    project_root = os.getcwd()  # Uses current directory
    print(f"Creating project structure in: {project_root}")
    create_project_structure(project_root)