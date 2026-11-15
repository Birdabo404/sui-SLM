#!/usr/bin/env python3
"""The command checker: which commands and flags in an answer its excerpts never show. Excerpts are real tldr pages
from the index, cut to the lines that matter."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.command_check import answer_uses, check, excerpts_seen

TAR = """tar (tldr common)
# tar

> Archiving utility.

- [c]reate a g[z]ipped archive and write it to a [f]ile:

`tar czf {{path/to/target.tar.gz}} {{path/to/file1 path/to/file2 ...}}`

- E[x]tract a (compressed) archive [f]ile into the current directory [v]erbosely:

`tar xvf {{path/to/source.tar.ext}}`

- E[x]tract a (compressed) archive [f]ile into the target directory:

`tar xf {{path/to/source.tar.ext}} {{[-C|--directory]}} {{path/to/directory}}`
"""

GREP = """grep (tldr common)
# grep

- Search for an exact string (disables `regex`es):

`grep {{[-F|--fixed-strings]}} "{{exact_string}}" {{path/to/file}}`

- Search for a pattern in all files recursively in a directory, ignoring binary files:

`grep {{[-rI|--recursive --binary-files=without-match]}} "{{search_pattern}}" {{path/to/directory}}`

- Print 3 lines of [C]ontext around, [B]efore, or [A]fter each match:

`grep {{--context|--before-context|--after-context}} 3 "{{search_pattern}}" {{path/to/file}}`

- Print file name and line number for each match with color output:

`grep {{[-Hn|--with-filename --line-number]}} --color=always "{{search_pattern}}" {{path/to/file}}`
"""

APT = """apt (tldr linux)
# apt

> Package manager for Debian-based distributions.
> Intended as a user-friendly alternative to `apt-get` for interactive use.

- Update the list of available packages and versions (recommended before running other `apt` commands):

`sudo apt update`

- Install a package, or update it to the latest version:

`sudo apt install {{package}}`

- List all installed packages:

`apt list {{[-i|--installed]}}`
"""

DPKG = """dpkg (tldr linux)
# dpkg

- List installed packages:

`dpkg {{[-l|--list]}} {{pattern}}`

- Install a package:

`sudo dpkg {{[-i|--install]}} {{path/to/file.deb}}`
"""

SSH = """ssh (tldr common)
# ssh

- Connect to a remote server with a specific [i]dentity (private key):

`ssh {{username}}@{{remote_host}} -i {{path/to/key_file}}`

- Connect to a remote server with IP `10.0.0.1` and using a specific [p]ort:

`ssh {{username}}@10.0.0.1 -p {{2222}}`
"""

IP = """ip (tldr linux)
# ip

- List interfaces with brief network layer info:

`ip {{[-br|-brief]}} {{[a|address]}}`
"""

FIND = """find (tldr common)
# find

- Find files by extension:

`find {{path/to/directory}} -name '{{*.ext}}'`

- Find directories matching a given name, in case-insensitive mode:

`find {{path/to/directory}} -type d -iname '{{*lib*}}'`

- Run a command for each file (use `{}` within the command to access the filename):

`find {{path/to/directory}} -name '{{*.ext}}' -exec {{wc -l}} {} \\;`
"""

CHMOD = """chmod (tldr common)
# chmod

- Give the [u]ser who owns a file the right to e[x]ecute it:

`chmod u+x {{path/to/file}}`

- Recursively give [a]ll users [r]ead permissions to files:

`chmod {{[-R|--recursive]}} a+rX {{path/to/directory}}`
"""

SYSTEMCTL = """systemctl (tldr linux)
# systemctl

- Start/Stop/Restart/Reload/Show the status of a service:

`systemctl {{start|stop|restart|reload|status}} {{unit}}`

- List all service/socket/automount units filtering by running/failed state:

`systemctl list-units {{[-t|--type]}} {{service|socket|automount|...}} --state {{failed|running}}`
"""

PS = """ps (tldr common)
# ps

- List all running processes:

`ps aux`
"""


def bash(*lines):
    return "Answer:\n```bash\n" + "\n".join(lines) + "\n```\nSource: [1]"


def commands(answer):
    return [(u.command, u.flags) for u in answer_uses(answer)]


def missing(answer, *pages):
    return check(answer, pages).missing


class ExtractTests(unittest.TestCase):
    def test_the_first_word_is_the_command_and_dash_words_are_its_flags(self):
        self.assertEqual(commands(bash("tar -xzf archive.tar.gz -C /tmp")), [("tar", ("-xzf", "-C"))])

    def test_sudo_and_its_own_options_are_skipped(self):
        self.assertEqual(commands(bash("sudo apt install -y openssh-server")), [("apt", ("-y",))])
        self.assertEqual(commands(bash("sudo -u postgres psql -c 'SELECT 1'")), [("psql", ("-c",))])
        self.assertEqual(commands(bash("sudo -E -H pip install x")), [("pip", ())])
        self.assertEqual(commands(bash("sudo -i", "sudo -s")), [])
        self.assertEqual(commands(bash("sudo -- rm -rf build")), [("rm", ("-rf",))])

    def test_other_prefixes_and_environment_assignments_are_skipped(self):
        self.assertEqual(commands(bash("DEBIAN_FRONTEND=noninteractive apt-get install -y curl")),
                         [("apt-get", ("-y",))])
        self.assertEqual(commands(bash("nice -n 10 make -j4")), [("make", ("-j4",))])
        self.assertEqual(commands(bash("env LANG=C sort file")), [("sort", ())])
        self.assertEqual(commands(bash("time nohup ./run.sh")), [])

    def test_each_part_of_a_pipeline_or_list_is_its_own_command(self):
        self.assertEqual(commands(bash("sudo apt update && sudo apt upgrade -y; reboot")),
                         [("apt", ()), ("apt", ("-y",)), ("reboot", ())])
        self.assertEqual(commands(bash("kill $(pgrep -f firefox)")), [("pgrep", ("-f",))])
        self.assertEqual(commands(bash("make || echo failed")), [("make", ())])

    def test_a_filter_reading_a_pipe_is_not_checked_but_leading_a_line_it_is(self):
        self.assertEqual(commands(bash("dpkg -l | grep -i nginx | head -n 5")), [("dpkg", ("-l",))])
        self.assertEqual(commands(bash("grep -rn TODO src/")), [("grep", ("-rn",))])
        self.assertEqual(commands(bash("echo 1 | sudo tee -a /etc/x")), [])

    def test_runners_give_both_themselves_and_the_command_they_run(self):
        self.assertEqual(commands(bash("find . -type f -print0 | xargs -0 -I {} grep -l TODO {}")),
                         [("find", ("-type", "-print0")), ("xargs", ("-I",)), ("grep", ("-l",))])
        self.assertEqual(commands(bash("timeout 5s ping -c 3 8.8.8.8")), [("timeout", ()), ("ping", ("-c",))])
        self.assertEqual(commands(bash("watch -n 1 df -h")), [("watch", ("-n",)), ("df", ("-h",))])
        self.assertEqual(commands(bash("find . -name '*.log' -exec rm -f {} \\;")),
                         [("find", ("-name", "-exec")), ("rm", ("-f",))])

    def test_placeholders_paths_and_variables_are_never_commands(self):
        self.assertEqual(commands(bash("<command> --help")), [])
        self.assertEqual(commands(bash("{{command}} --help")), [])
        self.assertEqual(commands(bash("./configure --prefix=/usr/local")), [])
        self.assertEqual(commands(bash("/usr/bin/python3 -V")), [])
        self.assertEqual(commands(bash("$EDITOR ~/.bashrc", "${SHELL} -l")), [])
        self.assertEqual(commands(bash("chmod 600 <path/to/file>")), [("chmod", ())])
        self.assertEqual(commands(bash("ssh -p 2222 <username>@<host>")), [("ssh", ("-p",))])

    def test_quoted_arguments_and_redirections_are_not_flags_or_commands(self):
        self.assertEqual(commands(bash('grep -e "-v" file')), [("grep", ("-e",))])
        self.assertEqual(commands(bash("echo 'rm -rf /' > notes.txt")), [])
        self.assertEqual(commands(bash("sort < in.txt > out.txt 2>&1")), [("sort", ())])
        self.assertEqual(commands(bash("awk '{print $1}' /etc/passwd")), [("awk", ())])
        self.assertEqual(commands(bash("unbalanced 'quote -x")), [("unbalanced", ())])

    def test_negative_numbers_and_the_end_of_options_are_not_flags(self):
        self.assertEqual(commands(bash("head -n -5 file", "ls -1", "kill -9 1234")),
                         [("head", ("-n",)), ("ls", ())])
        self.assertEqual(commands(bash("rm -- -weird-name")), [("rm", ())])

    def test_shell_builtins_and_shells_are_skipped(self):
        self.assertEqual(commands(bash("cd ~/project", "export PATH=$HOME/bin:$PATH", "source ~/.bashrc",
                                       "echo done", "bash install.sh", "sudo sh -c 'echo 1 > /proc/x'")), [])

    def test_prompts_comments_continuations_and_heredocs(self):
        self.assertEqual(commands(bash("# update first", "$ sudo apt update", "me@lab-pc:~$ ls -l")),
                         [("apt", ()), ("ls", ("-l",))])
        self.assertEqual(commands(bash("rsync -av \\", "  src/ dest/  # copy")), [("rsync", ("-av",))])
        self.assertEqual(commands(bash("cat <<EOF > /etc/x.conf", "foo -bar", "EOF", "systemctl restart x")),
                         [("cat", ()), ("systemctl", ())])

    def test_console_blocks_skip_output_lines(self):
        answer = "```console\n$ ls -l\ntotal 4\n-rw-r--r-- 1 me me 0 file\n```"
        self.assertEqual(commands(answer), [("ls", ("-l",))])

    def test_blocks_in_other_languages_are_not_read(self):
        self.assertEqual(commands("```python\nimport os\nos.system('rm -rf /')\n```"), [])
        self.assertEqual(commands("```ini\n[Service]\nExecStart=/usr/bin/foo -x\n```"), [])

    def test_inline_code_counts_only_when_it_reads_as_a_command_line(self):
        answer = ("Run `tar -xzf file.tar.gz`, then `ls`. Edit the `/etc/hosts` file, pass the `-r` flag, "
                  "press `Ctrl + C`, or see `man tar`.")
        self.assertEqual(commands(answer), [("tar", ("-xzf",)), ("man", ())])


class ExcerptTests(unittest.TestCase):
    def test_tldr_alternatives_placeholders_and_dashless_tar_bundles(self):
        tar, grep = excerpts_seen([TAR, GREP])
        self.assertIn("--directory", tar.flags)
        self.assertTrue({"c", "z", "f", "x", "v", "C"} <= tar.letters)
        self.assertTrue({"--fixed-strings", "--recursive", "--binary-files", "--context", "--after-context",
                         "--color"} <= grep.flags)
        self.assertTrue({"F", "r", "I", "H", "n"} <= grep.letters)
        self.assertNotIn("Search", grep.flags)


class CheckTests(unittest.TestCase):
    """Real answers against the pages the index returns for them."""

    def test_tar_with_dashed_flags_matches_the_dashless_page(self):
        self.assertEqual(missing(bash("tar -xzf archive.tar.gz", "tar -xvf a.tar -C /tmp"), TAR), ())

    def test_a_flag_letter_the_page_never_shows_is_named_on_its_own(self):
        self.assertEqual(missing(bash("tar -xjf archive.tar.bz2"), TAR), ("tar -j",))

    def test_combined_and_long_grep_flags(self):
        self.assertEqual(missing(bash('grep -rn "TODO" src/', "grep --context 3 x f", "grep -Fx word f"), GREP),
                         ("grep -x",))

    def test_apt_subcommands_pass_and_an_unshown_yes_flag_is_named(self):
        self.assertEqual(missing(bash("sudo apt update", "sudo apt install nginx"), APT), ())
        self.assertEqual(missing(bash("sudo apt install -y nginx"), APT), ("apt -y",))

    def test_an_invented_command_is_named(self):
        self.assertEqual(missing(bash("sudo apt-fast install nginx"), APT), ("apt-fast",))

    def test_a_piped_grep_is_fine_on_a_dpkg_page(self):
        self.assertEqual(missing(bash("dpkg -l | grep nginx", "sudo dpkg -i ./pkg.deb"), DPKG), ())

    def test_single_dash_long_options_are_whole_words(self):
        self.assertEqual(missing(bash("ip -br a"), IP), ())
        self.assertEqual(missing(bash("ip -c a"), IP), ("ip -c",))
        self.assertEqual(missing(bash("find . -iname '*.txt'", "find /var/log -type f -name '*.gz'"), FIND), ())
        self.assertEqual(missing(bash("find . -mmin -5"), FIND), ("find -mmin",))

    def test_ssh_with_placeholders(self):
        self.assertEqual(missing(bash("ssh -p 2222 <username>@<host>", "ssh -i ~/.ssh/id_ed25519 me@lab-pc"), SSH),
                         ())

    def test_chmod_long_alternative_and_an_unshown_long_flag(self):
        self.assertEqual(missing(bash("chmod -R 755 dir", "chmod --recursive a+r dir"), CHMOD), ())
        self.assertEqual(missing(bash("chmod --reference=a b"), CHMOD), ("chmod --reference",))

    def test_systemctl_with_sudo_and_a_value_flag(self):
        self.assertEqual(missing(bash("sudo systemctl restart ssh", "systemctl list-units --type service"),
                                 SYSTEMCTL), ())

    def test_ps_aux_matches_the_dashless_page(self):
        self.assertEqual(missing(bash("ps aux | grep firefox", "ps -aux"), PS), ())

    def test_a_flag_counts_when_an_excerpt_mentioning_the_command_shows_it(self):
        prose = "lab-notes (lab)\nUse `rsync -av` to copy, or pass -z to compress."
        self.assertEqual(missing(bash("rsync -avz src/ dest/"), prose), ())
        self.assertEqual(missing(bash("rsync -avP src/ dest/"), prose), ("rsync -P",))
        self.assertEqual(missing(bash("rsync -avz src/ dest/"), "rsync (lab)\nUse `rsync -av`.", "Pass -z to gzip."),
                         ("rsync -z",))

    def test_a_command_only_mentioned_in_prose_still_counts_as_shown(self):
        self.assertEqual(missing(bash("apt-get install curl"), APT), ())

    def test_each_item_is_listed_once_and_the_result_serializes(self):
        result = check(bash("tar -xjf a", "tar -xjf b", "apt-fast x", "apt-fast y"), [TAR, APT])
        self.assertEqual(result.missing, ("apt-fast", "tar -j"))
        self.assertFalse(result.ok)
        self.assertEqual(result.as_dict(), {"commands": ["tar", "apt-fast"], "missing": ["apt-fast", "tar -j"],
                                            "flags_checked": 2})

    def test_no_commands_or_no_excerpts(self):
        self.assertTrue(check("Answer: I don't know.", [TAR]).ok)
        self.assertEqual(check("Answer: plain text", []).commands, ())
        self.assertEqual(missing(bash("tar -xzf a"), ""), ("tar",))


if __name__ == "__main__":
    unittest.main()
