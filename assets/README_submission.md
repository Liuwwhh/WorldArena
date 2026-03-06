# WorldArena Evaluation and Submission Guide

This document outlines the procedure for evaluating your model on the WorldArena benchmark and submitting your results for the official leaderboard.

## 1. Data Preparation

First, download the test datasets from our official Hugging repository:
[WorldArena_Robotwin2.0](https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0)

You can download the following folders:
- `test_dataset`: Evaluation set for **Leaderboard**.
- `val_dataset`: Used for the **Online Arena** (head-to-head video comparison). This set allows you to upload your generated videos for a specific episode and compare them against existing baselines with real-time metrics.

Final evaluation results will be synchronized to `Leaderboard` and `Online Arena (optional)`, respectively.

***Notice:*** *The final leaderboard is evaluated on the **test_dataset**. The **val_dataset** is designated for Arena visualization; its inclusion in the submission is optional.*

### Inference Requirements
For each episode in the test/validation sets, use your model to generate a video based on the provided initial frame (`first_frame`) and text instruction (`instruction`) or actions(`data/_traj_data`).

- **Resolution**: Recommend to be **640×480** or higher.
- **Length**: Generate **121 frames**.
- **Frame Rate**: **24 fps**.

---

## 2. Submission Format

You need to package your generated videos and model information into a single archive (e.g., `.zip`, `.tar`).

### A) Archive Structure
Your submission should follow the format of our [example_eval.zip](https://huggingface.co/datasets/WorldArena/WorldArena_Robotwin2.0/blob/main/example_eval.zip). The archive should be named `{Your_Model_Name}_eval` and contain:
1. **Video Folders**: Separate folders for each evaluation set (e.g., `example_test(_1,_2)`, `example_arena(_1,_2)`).
2. **Model Documentation**: A `model_README.md` (or `.txt`) file.

### B) Model Documentation Details
The `model_README.md` should contain:
- **Model Name**
- **GitHub Repository (optional)** 
- **Driver Type**: Action-driven or Text-driven
- **Release Year**
- **Open Source Status**: Yes/No
- **Brief Description (optional)**
- **Communication methods (optional)**
- **Submission video set**: Specify which sets you are submitting (`example_test(_1,_2)`, `example_val(_1,_2)`, or `both`).
- **Other information (optional)**

---

## 3. Submission Process

### Step 1: Package Files
Ensure all videos and the `model_README.md` are correctly organized within the `{Your_Model_Name}_eval` archive.

### Step 2: Send Email
Email your archive to: **WorldArena1@outlook.com**
- **Subject**: `{Your_Model_Name}_evaluation`
- **Attachment**: `{Your_Model_Name}_eval.zip` 

---

## 4. Evaluation Cycle & Leaderboard Updates

- **Updates**: After your submission, the model will be evaluated and updated to the [WorldArena Leaderboard](https://huggingface.co/spaces/WorldArena/WorldArena) within **3–4 days**.
- **Notification**: We will send you a confirmation email once the evaluation is complete. Thank you for your patience and contribution!

