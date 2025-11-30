# Pipeline Environment Setup Guide

## Step 1: Install Java 11 (Temurin)

1. Download Java 11 (Temurin) from: https://adoptium.net/es/temurin/releases?os=any&package=jdk&version=11&arch=any
   - Select **Windows x64 JDK** installer

2. Run the installer and note the installation path: `C:\Program Files\Eclipse Adoptium\jdk-11-0.29.7-hotspot\`

3. Set environment variables:
   - Open **System Properties** → **Environment Variables**
   - Create new system variable:
     - Variable name: `JAVA_HOME`
     - Variable value: `C:\Program Files\Eclipse Adoptium\jdk-11-0.29.7-hotspot\`
   - Edit `Path` system variable and add: `%JAVA_HOME%\bin`

4. Verify installation (restart your terminal first):

```powershell
java -version
```

Expected output: `openjdk version "11.0.29"`

---

## Step 2: Install Scala 2.12.17

1. Download Scala 2.12.17 Windows installer from:
   https://downloads.lightbend.com/scala/2.12.17/scala-2.12.17.msi

2. Run the installer

3. Set environment variables:
   - Create system variable:
     - Variable name: `SCALA_HOME`
     - Variable value: `C:\Program Files (x86)\scala`
   - Add to `Path`: `%SCALA_HOME%\bin`

4. Verify installation:

```powershell
scala -version
```

Expected output: `Scala code runner version 2.12.17`

---

## Step 3: Install Apache Spark 3.4.1 with Hadoop 3.3.5

1. Download Spark 3.4.1 with Hadoop 3 from the Apache archive:
   https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz

2. Extract the `.tgz` file using tar tool built-in for windows in powershell:

```powershell
tar -xzf spark-3.4.1-bin-hadoop3.tgz
```

3. Move the extracted folder to: `C:\spark\spark-3.4.1-bin-hadoop3`

4. Download winutils.exe and hadoop.dll for Hadoop 3.3.5:
   - Download from: https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.5/bin
   - Place both `winutils.exe` and `hadoop.dll` in: `C:\spark\spark-3.4.1-bin-hadoop3\bin`

5. Set environment variables:
   - Create system variables:
     - Variable name: `SPARK_HOME`
     - Variable value: `C:\spark\spark-3.4.1-bin-hadoop3`
     - Variable name: `HADOOP_HOME`
     - Variable value: `C:\spark\spark-3.4.1-bin-hadoop3`
   - Add to `Path`: 
     - `%SPARK_HOME%\bin`
     - `%HADOOP_HOME%\bin`

6. Place JAR files:
   - Copy the JAR files from the GitHub repository into: `C:\spark\spark-3.4.1-bin-hadoop3\jars`

7. Verify installation (restart terminal):

```powershell
spark-submit --version
```

Expected output should show:
- Spark version 3.4.1
- Scala version 2.12.17
- Java version 11.0.29
- Hadoop version 3.3.5

---

## Step 4: Install MongoDB and Connector JARs

Install MongoDB Community Server (https://www.mongodb.com/try/download/community). 

Note: The MongoDB connector JAR files should be placed in `C:\spark\spark-3.4.1-bin-hadoop3\jars` (same location as other JARs from the GitHub repository). 

---

## Step 5: Install Git and Clone Repository

1. Download and install Git from: https://git-scm.com/install/

2. Follow the installation wizard setting the recommended settings.

3. After installation, open **PowerShell** or **Command Prompt** and navigate to your desired workspace (e.g., Desktop):

```powershell
cd Desktop
```

4. Clone the project repository:

```powershell
git clone https://github.com/lluc-palou/rdl-lob.git
```

5. Navigate into the cloned repository:

```powershell
cd rdl-lob
```

---

## Step 6: Install Anaconda

Sign in, download and install Anaconda:

- **Anaconda**: https://www.anaconda.com/download

Follow the installation wizard and set recommended settings. 

---

## Step 7: Create Conda Environment

1. Open **Anaconda PowerShell Prompt**

2. Navigate to the cloned repository:

```powershell
cd Desktop\rdl-lob
```

3. Navigate into the env directory and create the environment using `environment.yaml`:

```powershell
conda env create -f environment.yaml
```

Once created, activate the environment:

```powershell
conda activate tfg
```

---

## Step 8: Configure PySpark to Use Conda Environment

1. Activate your Conda environment:

```powershell
conda activate tfg
```

2. Set environment variables permanently:

```powershell
setx PYSPARK_PYTHON "C:\ProgramData\anaconda3\envs\tfg\python.exe"
setx PYSPARK_DRIVER_PYTHON "C:\ProgramData\anaconda3\envs\tfg\python.exe"
```

---

## Step 9: Verification

After completing all steps and restarting your terminal:

1. Activate your Conda environment:
```powershell
conda activate tfg
```

2. Launch PySpark:
```powershell
pyspark
```

---

## Step 10: Run and Verify MLflow

1. Ensure your Conda environment is activated:
```powershell
conda activate tfg
```

2. Navigate to your project directory:
```powershell
cd Desktop\rdl-lob
```

3. Start MLflow UI locally:
```powershell
mlflow ui
```

4. Open your web browser and navigate to:
```
http://localhost:5000
```

Expected result: You should see the MLflow tracking UI displaying experiments and runs.

5. To stop MLflow, press `Ctrl+C` in the terminal.
