# NEXUS-ML (Job Extraction and Job Matching using Resume and Job Posting)

## Setup Guide (Windows 11)

## Prerequisites

*   **Python 3.12**: Ensure Python 3.12 is installed and added to your PATH.
*   **NVIDIA GPU**: This project requires an NVIDIA GPU with CUDA support.

### NVIDIA CUDA Installation

Please download and install the NVIDIA CUDA Toolkit. Based on the project dependencies, **CUDA 13** is required.

*   [Download NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

## Installation Commands

Open your terminal (PowerShell or Command Prompt) and run the following commands in order:

```powershell
# 1. Create a virtual environment
python -m venv .env

# 2. Activate the virtual environment
.env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Updated Functions (Eleventh Commit)

### 1. ClamAV Virus Scanner (`resume_virus_scanner.py`)

ClamAV is now integrated as a mandatory pre-scan gate before any resume file enters the NLP pipeline. Every uploaded file is scanned locally — no data is sent to third-party servers.

#### Installation

```bash
sudo dnf install clamav clamd clamav-update
```

#### Unlock Config Files

```bash
sudo sed -i 's/^Example/#Example/' /etc/freshclam.conf
sudo sed -i 's/^Example/#Example/' /etc/clamd.d/scan.conf
```

#### Define the Socket

```bash
sudo bash -c 'echo "LocalSocket /run/clamd.scan/clamd.sock" >> /etc/clamd.d/scan.conf'
sudo mkdir -p /run/clamd.scan
sudo chown clamscan:clamscan /run/clamd.scan
```

#### Update Virus Definitions

```bash
sudo freshclam
```

#### Start the Daemon

```bash
sudo systemctl start clamd@scan
sudo systemctl enable clamd@scan
```

#### Verify Daemon is Running

```bash
sudo systemctl status clamd@scan
```

#### Confirm Socket Exists

```bash
sudo ls -la /run/clamd.scan/clamd.sock
```

#### Install Python Binding

```bash
pip install clamd
```

#### Set Socket Path

```bash
export CLAMAV_SOCKET=/run/clamd.scan/clamd.sock
echo 'export CLAMAV_SOCKET=/run/clamd.scan/clamd.sock' >> ~/.bashrc
source ~/.bashrc
```

#### Grant Your User Permission to Access the Socket

Add your user to the `clamscan` group and set the correct socket permissions:

```bash
sudo usermod -a -G clamscan $USER
sudo chmod 770 /run/clamd.scan
sudo chmod 770 /run/clamd.scan/clamd.sock
```

> **Important — group changes require a session restart.**
> Simply closing and reopening the terminal is not enough.
> Run the following command to apply the group change immediately without logging out:
>
> ```bash
> newgrp clamscan
> ```
>
> Then reactivate your virtual environment in the same terminal:
>
> ```bash
> source .env/bin/activate
> ```

#### Verify Python Can Connect

```bash
python3.12 -c "import clamd; cd = clamd.ClamdUnixSocket(path='/run/clamd.scan/clamd.sock'); print(cd.ping())"
```

Expected output: `PONG`

> **Note on permissions — why `770` and not `777`:**
> `770` allows only the owner and `clamscan` group members to access the socket.
> `777` allows everyone on the system, which is unnecessarily open.
> Use `770` — it is the correct and secure setting once your user is in the `clamscan` group.

#### Enable Auto-Update

```bash
sudo systemctl start freshclam
sudo systemctl enable freshclam
```

#### Stop the Daemon

```bash
# Temporarily
sudo systemctl stop clamd@scan

# Permanently
sudo systemctl stop clamd@scan
sudo systemctl disable clamd@scan
```

#### Uninstall

```bash
sudo systemctl stop clamd@scan
sudo systemctl disable clamd@scan
sudo systemctl stop freshclam
sudo systemctl disable freshclam
sudo dnf remove clamav clamd clamav-update
sudo rm -rf /etc/clamd.d
sudo rm -f /etc/freshclam.conf
sudo rm -rf /run/clamd.scan
pip uninstall clamd
sed -i '/CLAMAV_SOCKET/d' ~/.bashrc
source ~/.bashrc
```

---

### 2. SHA-256 Audit Trail (`resume_virus_scanner.py`, `resume_text_detector_cleaner.py`, `resume_image_ocr_cleaner.py`)

A SHA-256 hash of the raw resume text is computed and printed before the text is wiped from memory. This provides a tamper-evident audit trail — the hash proves the file was processed without retaining any of its content.

**How it works:**

- After text extraction and PII cleaning, the raw text is hashed using SHA-256 before being wiped from RAM.
- The hash is one-way — no PII or resume content can be reconstructed from it.
- It is printed to the console as a processing record.

**Example output:**

```
Raw text SHA-256 (audit trail): 0248b2726c2689b508eb467d10098fc36f70c50add6cbd8c1036db40fbbae567
```

No installation or configuration required. SHA-256 is part of Python's built-in `hashlib` module.

---

### 3. PDF and Parsed Text Memory Wipe (`resume_text_detector_cleaner.py`, `resume_image_ocr_cleaner.py`)

After processing, all in-memory strings and image arrays derived from the resume are securely overwritten before being released to the garbage collector. This prevents PII from lingering in RAM after the pipeline finishes.

**Two wipe functions are used:**

- `_wipe_str(s)` — overwrites the internal character buffer of a Python string with zeros using `ctypes.memset`. Used on raw text, cleaned text, and all intermediate text strings.
- `_wipe_ndarray(arr)` — overwrites a NumPy image array's data buffer with zeros in-place. Used on all OpenCV image arrays during OCR preprocessing.

**What gets wiped and when:**

| Data | Wiped After |
|---|---|
| Raw extracted text (full PII) | After SHA-256 hash is recorded |
| Cleaned text with structure | After being written to `.txt` file |
| Lowercased tokenized text | After being written to `.txt` file |
| Preprocessed extraction text | After being written to `.txt` file |
| Per-line spaCy doc | After each line is reconstructed |
| OpenCV image arrays (color, gray, denoised, binarized) | Immediately after each processing stage |
| Per-page OCR text | Immediately after being appended to the full text |

**On error paths:** If the pipeline exits due to an error (e.g. file is not a resume, unexpected exception), `raw_text` is wiped before `sys.exit()` is called so PII never lingers on a failed run.

No installation required. Uses Python built-ins `ctypes`, `gc`, and `numpy` (already a project dependency).

---

### 4. Skill and Knowledge Extraction Model — JobBERT (`resume_skill_knowledge_extractor.py`, `job_posting_skill_knowledge_extractor.py`)

Two JobBERT models are used to extract skills and technical knowledge from both the résumé and the job posting. Models are downloaded automatically from HuggingFace on first run and cached locally.

| Model | HuggingFace ID | Purpose |
|---|---|---|
| JobBERT Skill | `jjzha/jobbert_skill_extraction` | Extracts soft skills (e.g. communication, problem-solving) |
| JobBERT Knowledge | `jjzha/jobbert_knowledge_extraction` | Extracts technical knowledge (e.g. Python, Proteus, C++) |

**Models are saved locally on first run:**

```
Models/
├── jobbert_skill_extraction/
└── jobbert_knowledge_extraction/
```

**Output is saved as JSON:**

```
Extracted_skills_knowledge/
├── Resume_skills_knowledge.json
└── Job_posting_skills_knowledge.json
```

No manual download required. On first run the models are fetched automatically. Subsequent runs load from local storage.

---

### 5. Job Matching Model — CareerBERT (`career_similarity_matcher.py`)

CareerBERT computes a semantic similarity score between the résumé profile and the job posting profile. The score represents how well the applicant's background matches the job requirements.

| Model | HuggingFace ID | Purpose |
|---|---|---|
| CareerBERT | `lwolfrum2/careerbert-jg` | Computes cosine similarity between résumé and job posting |

**Model is saved locally on first run:**

```
Models/
└── careerbert-jg/
```

**Example output:**

```
JOB POSTING VS RESUME SIMILARITY

  Job Posting : Requires a Bachelor of Science in Electronics Engineering ...
  Resume      : BS Electronics Engineering. Experienced: Electronics Engineer ...

  Similarity Score: 0.8741
```

The score ranges from `0.0` (no match) to `1.0` (perfect match). No manual download required.