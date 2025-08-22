# installation instructions for BIGDL 
# do not run this file using bash command. Run each command separately in terminal
# to install conda on linux/windows/mac, use the following link
https://www.anaconda.com/docs/getting-started/miniconda/install#windows-command-prompt

# to activate conda, if not activated
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# check version to test if it is installed
conda --version


# Create conda environment
conda create -y -n bigdl_class python=3.7

# Activate conda environment
conda activate bigdl_class

# Install Java 8 (OpenJDK)
sudo apt-get update
sudo apt-get install -y openjdk-8-jre

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/' >> ~/.bashrc

# Install BigDL for Spark 3
pip install --pre --upgrade bigdl-spark3

# Install PyTorch (compatible version)
pip install torch==1.9.0 torchvision==0.10.0

