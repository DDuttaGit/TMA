# Demo Instructions

This directory contains a demo implementation using the RASP library. Please follow the steps below to set up the environment and run the example.

## Setup and Execution

1.  **Clone the RASP repository** inside this folder:
    ```bash
    git clone https://github.com/tech-srl/RASP
    ```

2.  **Navigate to the RASP directory**:
    ```bash
    cd RASP
    ```

3.  **Run the installation script** based on your operating system:
    * **Mac or Linux:** Run the setup script.
        ```bash
        ./setup.sh
        ```
    * **Windows:** Please refer to `windows instructions.txt` located in the repository.

4.  **Activate the virtual environment**:
    ```bash
    source raspenv/bin/activate
    ```

5.  **Move Demo Files**:
    ```bash
    mv ../U.py ../U_support.py ../U .
    ```

6.  **Run the Demo**:
    ```bash
    python U.py
    ```
