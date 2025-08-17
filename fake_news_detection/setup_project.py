import os

# Define the project structure
project_structure = {
    'fake_news_detection': {
        'files': ['main.py', 'web_app.py', 'requirements.txt', 'README.md'],
        'folders': {
            'data': ['sample_data.csv'],
            'models': ['saved_model.pkl'],
            'notebooks': ['analysis.ipynb']
        }
    }
}

def create_project_structure():
    # Create main directory
    main_dir = 'fake_news_detection'
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
        print(f"Created directory: {main_dir}")
    
    # Create files in main directory
    for file in project_structure[main_dir]['files']:
        file_path = os.path.join(main_dir, file)
        with open(file_path, 'w') as f:
            f.write('')  # Create empty file
        print(f"Created file: {file_path}")
    
    # Create subdirectories and their files
    for folder, files in project_structure[main_dir]['folders'].items():
        folder_path = os.path.join(main_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created directory: {folder_path}")
        
        for file in files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'w') as f:
                f.write('')  # Create empty file
            print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_project_structure()
    print("\nProject structure created successfully!")
    print("\nYour project structure:")
    print("fake_news_detection/")
    print("├── main.py")
    print("├── web_app.py")
    print("├── requirements.txt")
    print("├── data/")
    print("│   └── sample_data.csv")
    print("├── models/")
    print("│   └── saved_model.pkl")
    print("├── notebooks/")
    print("│   └── analysis.ipynb")
    print("└── README.md")