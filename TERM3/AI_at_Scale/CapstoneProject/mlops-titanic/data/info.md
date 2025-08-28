train.csv uis dvc tracked 

1. removed from git tracking by 


dvc init --subdir

dvc add data/raw/train.csv
git rm -r --cached 'data/raw/train.csv'
git commit -m "stop tracking data/raw/train.csv"

dvc add data/raw/train.csv
dvc config core.autostage true 
git add data/raw/train.csv.dvc .gitignore
git add data/raw/train.csv.dvc
git commit -m "Add Titanic dataset with DVC tracking"


remote data folder id : 18AsDgP3Z6G279Em2T03a34tkiXFVCF4G
https://drive.google.com/drive/folders/18AsDgP3Z6G279Em2T03a34tkiXFVCF4G?usp=drive_link


dvc remote add -d myremote gdrive://18AsDgP3Z6G279Em2T03a34tkiXFVCF4G
dvc remote modify myremote gdrive_use_service_account true  # optional, had we been using service accounts

---------------------------------------------------------- 
Gdrive not working since we need to use service account + setup custom app in google console 

Moving instead with Git LFS 

git lfs install -- this has to be run at the root of the dir 

git lfs track "*.csv" "*.pkl" "*.pt" "*.pth"

.gitattributes at the root tracks the file deltas uploaded by Git large file system 


