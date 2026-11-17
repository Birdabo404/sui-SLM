# Raw app-path Ollama stream diagnosis

Dataset: first 20 rows of `eval/dev.jsonl` (DEV, not held-out).

The fine-tuned thesis-student GGUF was unavailable. The student side is the matched public base GGUF under the historically risky plain {{ .Prompt }} template; it is not the real student.

Each request was built by the app's `_guarded_chat_body(app_messages(...))` path with `stream:true`, `think:false`, temperature 0.2, context 4096, prediction cap 512, repeat penalty 1.1, the app keep-alive setting, and no seed. One synthetic excerpt containing the DEV gold answer was supplied to keep this a generation/stream sanity check rather than a retrieval-quality check.

App options: `{"num_ctx": 4096, "num_predict": 512, "repeat_penalty": 1.1, "temperature": 0.2}`

The complete system prompt is saved beside this report. The app sends the same +RAG+CoT thesis prompt and formatting overlay to both model roles. The intended ChatML template renders `.System` and `.Messages`; a stale plain `{{ .Prompt }}` template does not explicitly render either field, so the deployed student's Modelfile still determines whether the model sees it.

## Does Ollama already send one sentence?

- **thesis-base-q4km reproduction:** 1/20 raw responses were one sentence; 0/20 were empty before UI parsing.
- **student stale-template proxy (not fine-tuned):** 0/20 raw responses were one sentence; 0/20 were empty before UI parsing.

This statement applies only to the exact model IDs recorded below. A proxy capture does not establish what an unavailable fine-tuned student emits.

## Capture the actual ThinkPad tags

From the repository root on the ThinkPad:

```bash
python docs/diagnostics/capture_ollama_streams.py --prefix student-output-actual
```

The defaults are `thesis-base-q4km` and `thesis-student`; the command fails instead of substituting another model when either tag is unavailable.

## Per-answer stream results

| Model | DEV | Tokens | Done | Chunks | Markdown | Citations | Abstain | Sentences | Content / thinking chars |
|---|---|---:|---|---:|---|---|---|---:|---:|
| thesis-base-q4km reproduction | 01 `tuning-tar` | 106 | stop | 106 | yes | [1] | no | 4 | 422 / 0 |
| thesis-base-q4km reproduction | 02 `tuning-ssh` | 123 | stop | 121 | yes | [1] | no | 5 | 573 / 0 |
| thesis-base-q4km reproduction | 03 `tuning-apt` | 73 | stop | 72 | yes | [1] | no | 4 | 305 / 0 |
| thesis-base-q4km reproduction | 04 `tuning-chmod` | 64 | stop | 64 | yes | [1] | no | 2 | 253 / 0 |
| thesis-base-q4km reproduction | 05 `tuning-chown` | 117 | stop | 109 | yes | [1] | no | 5 | 434 / 0 |
| thesis-base-q4km reproduction | 06 `tuning-df` | 161 | stop | 161 | yes | [1] | no | 9 | 673 / 0 |
| thesis-base-q4km reproduction | 07 `tuning-du` | 139 | stop | 139 | yes | [1] | no | 4 | 632 / 0 |
| thesis-base-q4km reproduction | 08 `tuning-kill` | 60 | stop | 59 | yes | [1] | no | 2 | 236 / 0 |
| thesis-base-q4km reproduction | 09 `tuning-top` | 65 | stop | 65 | yes | [1] | no | 1 | 276 / 0 |
| thesis-base-q4km reproduction | 10 `tuning-scp` | 195 | stop | 188 | yes | [1] | no | 10 | 760 / 0 |
| thesis-base-q4km reproduction | 11 `tuning-ip` | 140 | stop | 139 | yes | [1] | no | 7 | 566 / 0 |
| thesis-base-q4km reproduction | 12 `tuning-ping` | 131 | stop | 131 | yes | [1] | no | 8 | 475 / 0 |
| thesis-base-q4km reproduction | 13 `tuning-grep` | 142 | stop | 140 | yes | [1] | no | 8 | 603 / 0 |
| thesis-base-q4km reproduction | 14 `tuning-find` | 67 | stop | 66 | yes | [1] | no | 2 | 264 / 0 |
| thesis-base-q4km reproduction | 15 `tuning-adduser` | 121 | stop | 120 | yes | [1] | no | 7 | 528 / 0 |
| thesis-base-q4km reproduction | 16 `tuning-passwd` | 107 | stop | 107 | yes | [1] | no | 7 | 452 / 0 |
| thesis-base-q4km reproduction | 17 `tuning-lsblk` | 147 | stop | 147 | yes | [1] | no | 9 | 579 / 0 |
| thesis-base-q4km reproduction | 18 `tuning-systemctl` | 143 | stop | 139 | yes | [1] | no | 7 | 593 / 0 |
| thesis-base-q4km reproduction | 19 `tuning-unzip` | 137 | stop | 137 | yes | [1] | no | 4 | 549 / 0 |
| thesis-base-q4km reproduction | 20 `tuning-sudo` | 130 | stop | 129 | yes | [1] | no | 12 | 522 / 0 |
| student stale-template proxy (not fine-tuned) | 01 `tuning-tar` | 123 | stop | 123 | yes | [1] | no | 7 | 499 / 0 |
| student stale-template proxy (not fine-tuned) | 02 `tuning-ssh` | 110 | stop | 110 | yes | [1] | no | 8 | 433 / 0 |
| student stale-template proxy (not fine-tuned) | 03 `tuning-apt` | 118 | stop | 118 | yes | [1] | no | 7 | 526 / 0 |
| student stale-template proxy (not fine-tuned) | 04 `tuning-chmod` | 139 | stop | 139 | yes | [1] | no | 5 | 596 / 0 |
| student stale-template proxy (not fine-tuned) | 05 `tuning-chown` | 108 | stop | 108 | yes | [1] | no | 7 | 415 / 0 |
| student stale-template proxy (not fine-tuned) | 06 `tuning-df` | 82 | stop | 82 | yes | [1] | no | 3 | 341 / 0 |
| student stale-template proxy (not fine-tuned) | 07 `tuning-du` | 125 | stop | 125 | yes | none | no | 3 | 465 / 0 |
| student stale-template proxy (not fine-tuned) | 08 `tuning-kill` | 98 | stop | 98 | yes | none | no | 8 | 412 / 0 |
| student stale-template proxy (not fine-tuned) | 09 `tuning-top` | 117 | stop | 117 | yes | [1] | no | 8 | 477 / 0 |
| student stale-template proxy (not fine-tuned) | 10 `tuning-scp` | 128 | stop | 128 | yes | [1] | no | 6 | 518 / 0 |
| student stale-template proxy (not fine-tuned) | 11 `tuning-ip` | 77 | stop | 77 | yes | [1] | no | 2 | 317 / 0 |
| student stale-template proxy (not fine-tuned) | 12 `tuning-ping` | 74 | stop | 74 | yes | [1] | no | 2 | 282 / 0 |
| student stale-template proxy (not fine-tuned) | 13 `tuning-grep` | 126 | stop | 126 | yes | [1] | no | 9 | 557 / 0 |
| student stale-template proxy (not fine-tuned) | 14 `tuning-find` | 118 | stop | 118 | yes | [1] | no | 4 | 480 / 0 |
| student stale-template proxy (not fine-tuned) | 15 `tuning-adduser` | 119 | stop | 119 | yes | [1] | no | 7 | 504 / 0 |
| student stale-template proxy (not fine-tuned) | 16 `tuning-passwd` | 125 | stop | 125 | yes | [1] | no | 8 | 526 / 0 |
| student stale-template proxy (not fine-tuned) | 17 `tuning-lsblk` | 91 | stop | 91 | yes | [1] | no | 7 | 358 / 0 |
| student stale-template proxy (not fine-tuned) | 18 `tuning-systemctl` | 143 | stop | 143 | yes | [1] | no | 9 | 635 / 0 |
| student stale-template proxy (not fine-tuned) | 19 `tuning-unzip` | 98 | stop | 98 | yes | [1] | no | 4 | 358 / 0 |
| student stale-template proxy (not fine-tuned) | 20 `tuning-sudo` | 101 | stop | 101 | yes | [1] | no | 7 | 435 / 0 |

## Side-by-side raw content samples

| DEV question | Base stream content | Student stream content |
|---|---|---|
| 01 how do I extract a tar.gz file | Reasoning: The user wants to know how to extract a `tar.gz` file using the provided documentation. Excerpt [1] contains the specific command syntax for this operation. - Step 1: Open your terminal and navigate to the directory containing the archive if needed… | Reasoning: The user wants to know how to extract a tar.gz file using the provided documentation. Excerpt [1] explicitly shows the command `tar -xzf archive.tar.gz`, which matches the requirement for extracting a compressed tar file. Answer: ```bash tar -xzf a… |
| 02 connect to another lab PC over ssh | Reasoning: The question asks how to connect to another lab PC using SSH, which is a standard command-line operation described in the provided documentation. I need to extract the specific syntax for initiating an SSH connection from the text. - Use the `ssh`… | Reasoning: The question asks how to connect to another lab PC using SSH, which is a direct command from the provided documentation. I will use excerpt [1]. Answer: ```bash ssh &lt;username&gt;@&lt;host&gt; ``` Steps: 1. Replace `&lt;username&gt;` with your actual username on t… |
| 03 install a package from the terminal | Reasoning: The user wants to know how to install a package using the terminal based on the provided documentation excerpt. I will use the command shown in the first line of the excerpt. ```bash sudo apt install package ``` Answer: 1. Run `sudo apt install &lt;pa… | Reasoning: The question asks how to install a package using the terminal, which is directly described as an `apt` command in the provided documentation. Answer: ```bash sudo apt install &lt;package_name&gt; ``` 1. Open your terminal and enter your root password (if… |
| 04 make a script executable | Reasoning: The user wants to make a script executable using the `chmod` command. Excerpt [1] provides the specific syntax for adding execute permissions (`+x`) to a file or script, which is exactly what is needed. ```bash chmod +x script ``` Answer: 1 | Reasoning: The user wants to execute a command to make a script executable using the `tldr` documentation excerpt provided. This requires adding execute permissions to the file, which is done with the `chmod` utility and the plus-x flag. I will use the first… |
| 05 change the owner of a folder | Reasoning: The question asks how to change the owner of a folder, which corresponds directly to the command provided in the first excerpt. - 1. Open a terminal and run `sudo chown &lt;username&gt;:&lt;group&gt; &lt;path/to/folder&gt;` (replace `&lt;username&gt;` with your desired us… | Reasoning: The question asks how to change the owner of a folder, which corresponds directly to the command provided in the documentation excerpt. Answer: ```bash sudo chown &lt;username&gt;:&lt;group&gt; /path/to/folder ``` 1. Run `sudo` followed by the command `chown`.… |
| 06 check how much disk space is left | Reasoning: To find out how much disk space is remaining, you need to use the `df` command with the `-h` flag to display human-readable units. This will show the current usage and available space for each mounted filesystem. 1. Open a terminal on your Linux la… | Reasoning: The user wants to know available disk space using standard Linux commands provided in the documentation. Excerpt [1] contains the `df` command, which lists filesystems and their usage. To see human-readable output (like &quot;used&quot;, &quot;available&quot;, &quot;percen… |
| 07 find which folder is using the most space | Reasoning: To determine which folder uses the most space, you can use the `du` command to list disk usage for all directories. The `-h` flag displays human-readable sizes, and `--max-depth=1` limits the search to only one level deep (the current directory and… | Reasoning: - The goal is to identify the directory consuming the largest amount of disk space within a specific depth. - The `du` command calculates disk usage for directories, and the `-h` flag provides human-readable output. - To find the top folder, we can… |
| 08 stop a frozen program | Reasoning: The question asks how to stop a frozen program, which corresponds to using the `kill` command with its specific syntax for terminating processes by their ID. I will use excerpt [1]. Answer: ```bash kill &lt;PID&gt; ``` Source: [1] | Reasoning: The user wants to stop a frozen program, which corresponds to killing its process ID (PID). The provided documentation excerpt contains the command syntax for this operation. Answer: ```bash kill &lt;PID&gt; ``` 1. Identify the PID of the frozen program… |
| 09 see which processes use the most CPU | Reasoning: The user wants to identify the processes consuming the most CPU resources using the `top` command, based on the provided documentation excerpt. ```bash # Run the &#x27;top&#x27; command in the background so you can continue working while it runs sudo top -b… | Reasoning: To find out which processes are consuming the most CPU, you can run a command that lists all running processes and sorts them by their CPU usage percentage. The `top` utility is specifically designed for this purpose in Linux environments. Answer:… |
| 10 copy a file to another computer on the network | Reasoning: The question asks how to copy a file from one computer to another on a network using SCP. Excerpt [1] provides the exact command syntax for this operation, which is `scp &lt;file&gt; user@host:`. This matches the requirement of copying a file to another… | Reasoning: The question asks how to copy a file to another computer using SCP, which is described in the first line of the provided excerpt. - 1. Open your terminal and ensure you have the `scp` command available (usually pre-installed). - 2. Run the followin… |
| 11 show my IP address | Reasoning: The question asks for the user&#x27;s current IP address, which is typically displayed by running a specific command to list network interfaces and their associated addresses. Based on standard Linux networking documentation, this information is found i… | Reasoning: The user wants to see their current IP address using the `ip` command. I will use excerpt [1] which contains the `ip addr` command, as this is the standard way to display network interface information including the IP address on Linux systems. Answ… |
| 12 test if the internet connection works | Reasoning: The question asks how to test an internet connection using a command found in the documentation. Excerpt [1] provides the exact `ping` command syntax with options for count and target IP, which is suitable for this purpose. Answer: ```bash ping -c… | Reasoning: The question asks how to verify an internet connection using a command from the provided documentation. Excerpt [1] contains the `ping` utility and its usage example, which is the standard tool for testing connectivity. Answer: ```bash ping -c 3 8.… |
| 13 search for text inside files | Reasoning: The question asks how to search for text inside files using the provided documentation excerpt. Excerpt [1] shows a command that uses `grep` with an option `-r` and a pattern, which is the standard way to recursively search for lines containing spe… | Reasoning: The question asks how to search for text inside files using the provided documentation excerpt. Excerpt [1] explicitly shows the command `grep -r pattern .` which performs a recursive search of all files in the current directory and subdirectories,… |
| 14 find a file by name | Reasoning: The user wants to use the `find` command to locate a specific file based on its name. The provided documentation excerpt [1] contains the exact syntax for this operation using the `-name` option. Answer: ```bash find . -name &lt;filename&gt; ``` Source:… | Reasoning: The question asks how to use the `find` command to locate a specific file based on its name. Excerpt [1] provides the syntax for this operation using the `-name` option followed by the filename argument. Answer: ```bash find . -name &lt;filename&gt; ```… |
| 15 create a new user account | Reasoning: The question asks how to create a new user account, which corresponds directly to the command provided in the first excerpt of the documentation. Answer: ```bash sudo adduser &lt;username&gt; ``` 1. Open your terminal and ensure you have administrative p… | Reasoning: The question asks how to create a new user account, which corresponds directly to the command provided in the first excerpt of the documentation. Answer: ```bash sudo adduser &lt;username&gt; ``` 1. Open your terminal and ensure you have administrative p… |
| 16 change my password | Reasoning: The user wants to change their password using the `passwd` command, which is documented in excerpt [1]. Since this is a direct command usage question, I will provide the steps to execute it. Answer: 1. Open your terminal and ensure you are logged i… | Reasoning: The user wants to change their password using the `passwd` command, which is documented in excerpt [1]. I will provide a short intro line followed by numbered steps for changing the password. ```bash sudo passwd &lt;username&gt; ``` ```bash echo $USER |… |
| 17 list USB drives and partitions | Reasoning: The user wants to see all mounted USB drives and their partitions using `lsblk`. This command lists block devices, including removable media like USBs. I will use the provided excerpt [1] which contains the exact command for this purpose. Answer: `… | Reasoning: The user wants to see a list of all USB drives and their partitions using `lsblk`. I will use the provided excerpt which shows how to run the command. Answer: 1. Open terminal (if not already open). 2. Run the following command in the terminal: ```… |
| 18 restart a service | Reasoning: The user wants to know how to restart a service using the provided documentation excerpt. Excerpt [1] explicitly shows the command syntax for restarting a service named &quot;name&quot;. Answer: ```bash sudo systemctl restart &lt;service_name&gt; ``` Steps: 1. Rep… | Reasoning: The question asks how to restart a service using the provided documentation excerpt. Excerpt [1] explicitly shows the command syntax for restarting a service named &quot;name&quot; with `sudo systemctl restart name`. This matches the user&#x27;s request exactly w… |
| 19 unzip a zip file | Reasoning: The user wants to know how to use the `unzip` command to extract files from a `.zip` archive based on the provided documentation. - Step 1: Open your terminal or command prompt in the directory where you want the files extracted. - Step 2: Type the… | Reasoning: The user wants to know how to use the `unzip` command to extract files from a `.zip` archive based on the provided documentation. - Step 1: Open your terminal or command prompt. - Step 2: Type the command `unzip file.zip`. - Step 3: Press Enter to… |
| 20 run a command as administrator | Reasoning: The question asks how to run a command as an administrator using the `sudo` tool, which is described in the provided documentation excerpt. I will extract the steps from this source. Answer: 1. Open your terminal or command prompt. 2. Type `sudo` a… | Reasoning: The user wants to know how to run a command with elevated privileges using the `sudo` utility, based on the provided documentation excerpt. I will extract the specific steps and commands from the text. Answer: 1. Open your terminal or command promp… |
