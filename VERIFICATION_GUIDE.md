# Repository Verification Guide

**Use this guide to self-verify the repository before public release.**

Since you are releasing code for a paper, it is critical that the repository works "out of the box" for other researchers. This checklist will help you verify that everything is clean and portable.

## 🛑 Pre-Release Checklist

1.  **[ ] Data Paths**:
    *   Check `food_scanner/config.yaml`: Ensure `input_dir`, `menu_json`, and other paths are relative (e.g., `./data/...`) or point to placeholders, NOT `/home/nutrition/...`.
    *   Check `weight_estimation/train.sh` and `evaluate.sh`: Verify that `DATA_ROOT` and other variables are set to placeholders users can understand.

2.  **[ ] Environment**:
    *   Creates a fresh conda environment and try installing requirements:
        ```bash
        conda create -n test_release python=3.10
        conda activate test_release
        pip install -r requirements.txt
        ```
    *   Does it install without errors? If `mmsegmentation` fails, ensure the README instructions for `mim install` are clear.

3.  **[ ] Sanity Check Scripts**:
    *   **Segmentation**: Run `python segmentation/train.py --help`. Does it run or crash due to missing imports?
    *   **Food Scanner**: Run `python food_scanner/src/app.py` (or verify imports work).
        *   *Note:* The app requires model checkpoints. Ensure `weight_model.py` is correctly imported (we fixed this to use relative imports, so it should work).

4.  **[ ] Sensitive Data**:
    *   Did you remove all WandB API keys? (We scanned for this, but double-check).
    *   Are there any internal server IPs in comments?

## 🧪 Verification Commands

Run these commands from the **root** of the repository:

### 1. Check Directory Structure
```bash
ls -F
# Should see:
# food_scanner/
# segmentation/
# weight_estimation/
# README.md
# requirements.txt
# LICENSE
```

### 2. Verify Config Sanitization
Run this grep command. **It should return NO output.**
```bash
grep -r "/home/nutrition" .
```
*(If it returns output in `__pycache__` or binary files, that's fine. It should not return output in `.py`, `.sh`, or `.yaml` files.)*

### 3. Test Imports
Try to import the main app module to ensure no missing dependencies.
```bash
cd food_scanner
python -c "from src.app import app; print('App imported successfully')"
```

---

## 📦 Packaging

When you are ready, you can simply zip this folder or push it to your new GitHub repository.

```bash
git init
git add .
git commit -m "Initial release for ICVGIP 2025"
git remote add origin <your-new-repo-url>
git push -u origin main
```
