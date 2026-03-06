# Chapter 1: Setting Up Your Environment

> No code to write yet. Just getting your computer ready.

---

## What You Need (and Why)

Before you can build anything, you need four tools installed on your computer:

1. **Python** - the programming language all the code in this book is written in.
2. **Git** - a tool for downloading the project code from the internet.
3. **PyTorch** - a math library that makes training neural networks fast and easy.
4. **A terminal** - the text-based window where you type commands to run code.

Think of it like setting up a kitchen before you cook. Python is the kitchen itself, PyTorch is a specialized appliance (a very powerful blender for numbers), Git is the service that delivers the recipe book to your door, and the terminal is how you talk to all of them.

You only have to do this setup once. After that, you just open the terminal and start coding.

---

## Step 1: Open a Terminal

Before you install anything, you need to know how to open the command window where you will type instructions.

**Windows:**

Press `Win + R`, type `cmd`, and press Enter. A black window appears with a blinking cursor. That is the Command Prompt - your terminal on Windows. Everything you type here gets executed immediately when you press Enter.

**macOS:**

Press `Cmd + Space` to open Spotlight, type `Terminal`, and press Enter. A white or dark window opens with a `$` prompt. That is the Terminal on macOS.

You will use this window for every step below.

---

## Step 2: Install Python

Python is the programming language this entire tutorial is written in. We need version 3.10 or newer.

**Windows:**

1. Open your web browser and go to **python.org/downloads**
2. Click the big yellow button: "Download Python 3.13.x" (the exact number after 3.13 does not matter)
3. Run the downloaded `.exe` installer
4. **CRITICAL:** On the first screen of the installer, check the box that says **"Add Python to PATH"** at the very bottom. If you miss this, Python will not be found when you type commands. Check it before clicking Install Now.
5. Click "Install Now" and wait for it to finish.

To verify it worked, open a new Command Prompt window and type:

```bash
python --version
```

You should see something like `Python 3.13.2`. If you do, Python is installed correctly.

**macOS:**

1. Open your web browser and go to **python.org/downloads** (same site)
2. Click the big yellow button: "Download Python 3.13.x"
3. Run the downloaded `.pkg` installer and follow the prompts (click Continue through all the screens)

To verify, open Terminal and type:

```bash
python3 --version
```

You should see `Python 3.13.x`. On macOS, the command is `python3` (with the 3) instead of just `python`.

**Troubleshooting:**

- **Windows: "'python' is not recognized as an internal or external command"** - This means you forgot to check the "Add Python to PATH" box during installation. Uninstall Python from Control Panel, reinstall it, and this time check that box.
- **macOS: "command not found: python3"** - Try typing `python --version` instead (without the 3). If that also fails, retry the installation steps above.

---

## Step 3: Install Git

Git is a version control tool. For our purposes, it is simply the command we use to download the project from the internet.

**Windows:**

1. Go to **git-scm.com/download/win**
2. The download should start automatically. If not, click the link for "64-bit Git for Windows Setup"
3. Run the downloaded installer
4. Click "Next" through every screen - all the default settings are fine. You do not need to change anything.
5. Click "Install" at the end.

To verify, open a **new** Command Prompt window (close the old one and open a fresh one) and type:

```bash
git --version
```

You should see something like `git version 2.48.1.windows.1`.

**macOS:**

Git is often already installed on macOS. Open Terminal and type:

```bash
git --version
```

If Git is installed, you will see a version number. If it is not installed, macOS will pop up a dialog asking if you want to install the "Command Line Developer Tools" - click Install and wait for it to finish.

---

## Step 4: Download the Project

Now that Git is installed, you can download the project code with a single command.

A **repository** (or "repo") is just a folder of code stored online. The `git clone` command creates a copy of that folder on your computer.

Open your terminal and type:

```bash
git clone https://github.com/jackluucoding/build-llm-from-zero.git
```

Then move into the project folder:

```bash
cd build-llm-from-zero
```

You should now be inside the project folder. You can verify this by typing `dir` (Windows) or `ls` (macOS) to see the files - you should see folders like `book`, `src`, and `checkpoints`.

**Windows tip:** You can paste commands into Command Prompt by right-clicking inside the window.

---

## Step 5: Create a Virtual Environment

Here is a problem you will eventually run into without this step: different projects need different versions of the same library. Project A needs PyTorch version 1.0, Project B needs version 2.0. If you install both globally, they overwrite each other.

A **virtual environment** solves this by giving each project its own isolated box of installed packages. Think of it as each project having its own dedicated toolbox. Project A keeps its tools in its own toolbox. Project B keeps its tools in its own toolbox. They never interfere with each other.

Make sure your terminal is inside the `build-llm-from-zero` folder (the `cd` command in Step 4 should have taken you there), then run:

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

After you run the activate command, you will see `(venv)` appear at the start of your terminal prompt, like this:

```
(venv) C:\Users\yourname\build-llm-from-zero>
```

That `(venv)` tag tells you the virtual environment is active and ready. Every package you install from now on goes into this isolated box, not your main Python installation.

**Important:** Every time you open a new terminal window to work on this project, you need to run the activate command again. The `(venv)` tag disappears when you close the terminal. It does not install anything new - it just switches your active environment back on.

---

## Step 6: Install PyTorch and Dependencies

With the virtual environment active (you should see `(venv)` in your prompt), install the required libraries.

**pip** is Python's package manager - think of it as an app store for code libraries. When you type `pip install something`, it downloads and installs that library automatically.

First, install PyTorch. We use a CPU-only version because this tutorial does not require a graphics card:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

This downloads about 200 MB, so it may take a few minutes depending on your internet speed.

Then install the other libraries:

```bash
pip install numpy requests matplotlib
```

- **numpy**: math utilities used alongside PyTorch
- **requests**: for downloading the Shakespeare dataset
- **matplotlib**: for plotting training loss (used in one chapter)

To verify PyTorch installed correctly:

```bash
python -c "import torch; print(torch.__version__)"
```

You should see `2.6.0+cpu`. If you see that, PyTorch is ready.

**macOS note:** Use `python3` instead of `python` if the command is not found.

---

## Step 7: Download the Shakespeare Dataset

The tutorial trains the model on the complete works of Shakespeare, a text file about 1 MB in size. Run this to download it:

```bash
python src/utils/download_data.py
```

The script will print a confirmation message and show the first few lines of the text so you can see it worked. The file is saved to `src/data/shakespeare.txt`.

---

## Step 8: Verify Everything Works

Run the setup verification script to confirm all pieces are in place:

```bash
python src/ch00_setup_check.py
```

If everything is set up correctly, you will see:

```
Checking your environment...

[OK] Python 3.13.x
[OK] PyTorch 2.6.0+cpu
[OK] NumPy x.x.x
[OK] Requests x.x.x
[OK] Shakespeare dataset found (1,115,394 characters)
[OK] Quick tensor test passed

All checks passed! You are ready to start Chapter 1.
```

If any line shows `[FAIL]`, read the message next to it - it will tell you exactly which step to redo.

---

## Troubleshooting

**"pip: command not found" or "pip is not recognized"**
Try `pip3` instead of `pip`. If neither works, try `python -m pip install ...` (Windows) or `python3 -m pip install ...` (macOS).

**"ModuleNotFoundError: No module named 'torch'" when running a script**
Your virtual environment is not active. Open a terminal, navigate to the project folder, and run the activate command again (`venv\Scripts\activate` on Windows, `source venv/bin/activate` on macOS). You need to do this every time you open a new terminal window.

**Dataset download fails with a network error**
Check your internet connection. If the error persists, wait a few minutes and try again. The dataset can also be found by searching for "Tiny Shakespeare Karpathy" - download the raw text file and save it manually to `src/data/shakespeare.txt`.

**"python: command not found" on Windows even after installation**
The PATH checkbox was not checked during installation. Go to Control Panel > Apps, find Python, uninstall it, then reinstall and check the PATH box.

---

## Key Takeaways

- You installed four things: Python (the language), Git (code downloader), PyTorch (math library), and a terminal (how you talk to them).
- A virtual environment keeps this project's packages isolated from everything else on your computer.
- Always activate your virtual environment (`venv\Scripts\activate` or `source venv/bin/activate`) before working on this project.
- If `ch00_setup_check.py` shows all [OK], you are fully ready to begin.

---

*Next up: [Chapter 2, What Is a Language Model?](ch02-what-is-an-llm.md)*
