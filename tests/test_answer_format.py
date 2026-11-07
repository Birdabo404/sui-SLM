#!/usr/bin/env python3
"""The app's answer format: prompt overlay, parsing, placeholders, step follow-ups. No model is called."""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import shared.prompt as prompt
from app.answer_format import (
    app_messages,
    compare_step_note,
    is_inline_command,
    memory_text,
    normalize_placeholders,
    parse_answer,
    safe_markdown,
    side_label,
    step_note,
    step_query,
    step_reference,
)
from app.engine import MODES, build_messages

HITS = [{"source": "tldr", "title": "adduser", "path": "tldr/adduser.md", "text": "Create a user: adduser x"},
        {"source": "tldr", "title": "usermod", "path": "tldr/usermod.md", "text": "Add to group: usermod -aG g x"}]

SUDO = (
    "Create the account, then add it to the sudo group.\n\n"
    "1. Create the user and set a password:\n"
    "   ```bash\n   sudo adduser {{username}}\n   ```\n"
    "1. Add the user to the `sudo` group:\n"
    "   ```bash\n   sudo usermod -aG sudo {{username}}\n   ```\n"
    "1. Log in as the user and check:\n"
    "   ```bash\n   groups {{username}}\n   ```\n\n"
    "Source: [1], [2]"
)


class OverlayTests(unittest.TestCase):
    def test_eval_prompts_are_untouched(self):
        frozen = {name: copy.deepcopy(getattr(prompt, name)) for name in
                  ("SYSTEM", "SYSTEM_BASE", "SYSTEM_COT", "SYSTEM_RAG", "ABSTAIN")}
        before = {mode: build_messages("q", HITS, mode) for mode in MODES}
        for mode in MODES:
            app_messages("q", HITS, mode, step_note="note")
        self.assertEqual({mode: build_messages("q", HITS, mode) for mode in MODES}, before)
        self.assertEqual({name: getattr(prompt, name) for name in frozen}, frozen)

    def test_overlay_extends_the_thesis_system_prompt(self):
        for mode in MODES:
            system, user = app_messages("How do I add a user?", HITS, mode)
            self.assertTrue(system["content"].startswith(prompt.system_for_mode(mode)))
            self.assertIn("numbered steps", system["content"])
            self.assertIn("```bash", system["content"])
            self.assertIn("Do not turn it into steps", system["content"])
            self.assertNotEqual([system, user], build_messages("How do I add a user?", HITS, mode))

    def test_only_retrieval_modes_may_cite(self):
        for mode in ("base", "cot"):
            system, user = app_messages("q", HITS, mode)
            self.assertIn("do not write a Source line", system["content"])
            self.assertEqual(user["content"], "Question: q")
        for mode in ("rag", "full"):
            system, user = app_messages("q", HITS, mode)
            self.assertIn("Source: [n]", system["content"])
            self.assertEqual(user["content"], prompt.user_msg("q", HITS))

    def test_only_cot_modes_ask_for_reasoning(self):
        for mode in ("cot", "full"):
            self.assertIn("line Reasoning:", app_messages("q", HITS, mode)[0]["content"])
        for mode in ("base", "rag"):
            self.assertIn("Do not write a reasoning section", app_messages("q", HITS, mode)[0]["content"])

    def test_a_step_note_follows_the_question(self):
        user = app_messages("explain step 2", HITS, "full", step_note="NOTE")[1]["content"]
        self.assertTrue(user.endswith("\n\nNOTE"))


class PlaceholderTests(unittest.TestCase):
    def test_tldr_placeholders_become_angle_brackets(self):
        self.assertEqual(normalize_placeholders("sudo adduser {{username}}"), "sudo adduser <username>")
        self.assertEqual(normalize_placeholders("cp {{path/to/file}} {{dir}}"), "cp <path/to/file> <dir>")

    def test_other_brace_syntax_is_left_alone(self):
        for text in (
            "{{ item }}",
            "${{ secrets.TOKEN }}",
            "${{x}}",
            "docker ps --format '{{.Names}}'",
            "{{[-a|--all]}}",
            '{"a": {"b": 1}}',
            "echo {{{x}}}",
            "awk '{print $1}'",
            "echo {a,b}{{c}",
        ):
            self.assertEqual(normalize_placeholders(text), text, text)


class ParseTests(unittest.TestCase):
    def test_lazy_numbering_and_indented_fences(self):
        answer = parse_answer(SUDO, "full", n_hits=2)
        self.assertEqual([s.number for s in answer.steps], [1, 2, 3])
        self.assertEqual([s.commands for s in answer.steps],
                         [("sudo adduser <username>",), ("sudo usermod -aG sudo <username>",), ("groups <username>",)])
        self.assertEqual([b.kind for b in answer.blocks], ["md", "md", "code", "md", "code", "md", "code"])
        self.assertTrue(all(b.lang == "bash" for b in answer.blocks if b.kind == "code"))
        self.assertTrue(answer.blocks[3].text.startswith("2. Add the user"))
        self.assertEqual(answer.cited, (1, 2))
        self.assertTrue(answer.source_line)
        self.assertNotIn("Source", canonical(answer))
        self.assertNotIn("{{", canonical(answer))

    def test_step_headings_count_but_prose_about_a_step_does_not(self):
        steps = parse_answer("### Step 1: Make it\n```bash\nmkdir d\n```\n**Step 2 -** Enter it\n```\ncd d\n```").steps
        self.assertEqual([(s.number, s.text, s.commands) for s in steps],
                         [(1, "Make it", ("mkdir d",)), (2, "Enter it", ("cd d",))])
        self.assertEqual(parse_answer("Step 2 runs `usermod`.").steps, ())

    def test_nested_items_stay_in_their_step(self):
        steps = parse_answer("1. Edit the file:\n   1. open it\n   2. save it\n2. Reload").steps
        self.assertEqual([s.number for s in steps], [1, 2])

    def test_reasoning_is_split_only_on_an_answer_line_in_cot_modes(self):
        text = "Reasoning:\n- adduser makes it\n- usermod adds sudo\n\nAnswer:\n" + SUDO
        full = parse_answer(text, "full", n_hits=2)
        self.assertEqual(full.reasoning, "- adduser makes it\n- usermod adds sudo")
        self.assertEqual(len(full.steps), 3)
        self.assertEqual(parse_answer(text, "rag", n_hits=2).reasoning, "")
        self.assertEqual(parse_answer(SUDO, "full", n_hits=2).reasoning, "")

    def test_a_labelled_lead_without_answer_is_reasoning_up_to_the_first_step(self):
        # qwen3.5:2b's +CoT reply to the sudo question had this shape: no Answer: line, bullets numbered "- 1.".
        text = ("Reasoning: adduser makes the account, usermod adds the group.\n"
                "- 1. Create the new user account:\n```bash\nsudo adduser <username>\n```\n"
                "- 2. Add the user to the `sudo` group:\n```bash\nsudo usermod -aG sudo <username>\n```")
        answer = parse_answer(text, "cot")
        self.assertEqual(answer.reasoning, "adduser makes the account, usermod adds the group.")
        self.assertEqual([(s.number, s.commands) for s in answer.steps],
                         [(1, ("sudo adduser <username>",)), (2, ("sudo usermod -aG sudo <username>",))])
        self.assertTrue(answer.blocks[0].text.startswith("1. Create"))
        plain = parse_answer("Reasoning: it depends on the group.", "cot")
        self.assertEqual((plain.reasoning, plain.blocks[0].text), ("", "Reasoning: it depends on the group."))

    def test_a_procedure_above_answer_is_never_hidden(self):
        text = SUDO.replace("Source: [1], [2]", "Answer: that is all.")
        answer = parse_answer(text, "cot")
        self.assertEqual(answer.reasoning, "")
        self.assertEqual(len(answer.steps), 3)
        self.assertEqual(parse_answer("Some thoughts.\n\nAnswer:", "cot").reasoning, "")

    def test_citations_need_a_retrieved_excerpt(self):
        rag = parse_answer("Use `ls -l [3]` here [1] and [4].\n\nSource: [1], [9]", "rag", n_hits=3)
        self.assertEqual(rag.blocks[0].text, "Use `ls -l [3]` here [1] and.")
        self.assertEqual((rag.cited, rag.dropped), ((1,), (4, 9)))
        base = parse_answer("Use ls [1].\n\nSource: [1]", "base", n_hits=5)
        self.assertEqual(base.blocks[0].text, "Use ls.")
        self.assertEqual((base.cited, base.dropped), ((), (1,)))

    def test_streaming_shows_an_open_fence_and_holds_back_a_partial_line(self):
        partial = "Create it.\n\n1. Create the user:\n   ```bash\n   sudo adduser {{username}}\n   ``"
        answer = parse_answer(partial, "rag", n_hits=2, streaming=True)
        self.assertEqual(answer.blocks[-1].kind, "code")
        self.assertTrue(answer.blocks[-1].open)
        self.assertEqual(answer.blocks[-1].text, "sudo adduser <username>")
        held = parse_answer("Create it.\n\n1. Create the user:\n2", "rag", streaming=True)
        self.assertEqual(held.blocks[-1].text, "1. Create the user:")
        self.assertEqual(parse_answer("Create it.\n``", "rag", streaming=True).blocks[-1].kind, "md")

    def test_streaming_reasoning_waits_for_answer(self):
        early = parse_answer("Reasoning:\n- adduser makes the account", "cot", streaming=True)
        self.assertTrue(early.reasoning_open)
        self.assertEqual(early.blocks, ())
        later = parse_answer("Reasoning:\n- x\n\nAnswer:\nRun `id` to check it now.\n", "cot", streaming=True)
        self.assertFalse(later.reasoning_open)
        self.assertEqual(later.reasoning, "- x")

    def test_safe_markdown_shows_angle_brackets_outside_code(self):
        self.assertEqual(safe_markdown("Replace <username> in `id <username>`"),
                         "Replace &lt;username> in `id <username>`")

    def test_memory_text_is_the_shown_answer(self):
        text = "Reasoning:\n- private\n\nAnswer:\n" + SUDO
        memory = memory_text(text, "full")
        self.assertNotIn("private", memory)
        self.assertNotIn("Source", memory)
        self.assertNotIn("[1]", memory)
        self.assertIn("2. Add the user to the `sudo` group:\n```bash\nsudo usermod -aG sudo <username>\n```", memory)


class CommandPlacementTests(unittest.TestCase):
    # Shapes qwen3.5:2b wrote for the sudo question (+RAG, +CoT), cut down.
    RAG_INLINE = ("1. Create the new user using `adduser`.\n"
                  "2. Add the user to the `sudo` group using `usermod -aG sudo {{username}}`.\n"
                  "3. Edit the `/etc/sudoers` file, then run `visudo` and add `ALL=(ALL) NOPASSWD: ALL`.\n\n"
                  "Source: [1], [2]")
    COT_TRAILING = ("1. Create the new user account.\n2. Add the user to the sudo group.\n"
                    "3. Set up a home directory if it is missing.\n\n"
                    "```bash\nsudo adduser <username>\n```\n\n```bash\nsudo usermod -aG sudo <username>\n```\n\n"
                    "```bash\nsudo mkdir -p /home/<username>\n```")

    def test_inline_command_rule(self):
        for span in ("sudo usermod -aG sudo <username>", "usermod -aG sudo <username>", "apt update",
                     "groups <username>", "sudo reboot", 'tar -xzf "my file.tar.gz"', "cat ~/notes"):
            self.assertTrue(is_inline_command(span), span)
        for span in ("adduser", "visudo", "/etc/sudoers", "ALL=(ALL) NOPASSWD: ALL", "-aG sudo", "sudo group",
                     "the user", "file permissions", "id alice", 'echo "open', "sudo -u x ls", "Usermod -aG x y",
                     " ls -l"):
            self.assertFalse(is_inline_command(span), span)

    def test_a_whole_inline_command_gets_a_box_under_its_step_byte_for_byte(self):
        answer = parse_answer(self.RAG_INLINE, "rag", n_hits=2)
        codes = [(b.step, b.text, b.derived) for b in answer.blocks if b.kind == "code"]
        self.assertEqual(codes, [(2, "usermod -aG sudo <username>", True)])
        self.assertEqual(answer.blocks[1].text, "2. Add the user to the `sudo` group using `usermod -aG sudo <username>`.")
        self.assertEqual(answer.steps[1].commands, ("usermod -aG sudo <username>",))
        self.assertEqual(answer.steps[0].commands, ())
        quoted = parse_answer('1. Extract it with `tar -xzf "my file.tar.gz"`.').steps[0].commands
        self.assertEqual(quoted, ('tar -xzf "my file.tar.gz"',))

    def test_inline_commands_are_not_lifted_outside_steps_or_beside_fences(self):
        prose = parse_answer("Run `sudo usermod -aG sudo <username>`, then log in again.")
        self.assertEqual([b.kind for b in prose.blocks], ["md"])
        fenced = parse_answer("1. Add it with `usermod`:\n```bash\nsudo usermod -aG sudo x\n```\n"
                              "2. Check with `groups x`.\n```bash\ngroups x\n```")
        self.assertFalse(any(b.derived for b in fenced.blocks))

    def test_commands_after_steps_without_their_own_get_no_step(self):
        answer = parse_answer(self.COT_TRAILING, "cot")
        self.assertEqual([s.commands for s in answer.steps], [(), (), ()])
        self.assertEqual(answer.loose, ("sudo adduser <username>", "sudo usermod -aG sudo <username>",
                                        "sudo mkdir -p /home/<username>"))
        self.assertTrue(all(b.step is None for b in answer.blocks if b.loose))
        live = parse_answer(self.COT_TRAILING, "cot", streaming=True)
        self.assertEqual(live.loose, ())

    def test_commands_after_the_last_step_stay_with_it_when_the_layout_says_so(self):
        cases = {
            "only step": "1. Create the user:\n\n```bash\nsudo adduser x\n```\n\n```bash\nsudo passwd x\n```",
            "earlier steps fenced": ("1. Create:\n```bash\nsudo adduser x\n```\n2. Check:\n\n```bash\ngroups x\n```"
                                     "\n\n```bash\nid x\n```"),
            "right under it": "1. Update.\n2. Install it.\n```bash\nsudo apt install git\n```",
            "after a colon": "1. Update.\n2. Install it:\n\n```bash\nsudo apt install git\n```",
        }
        for name, text in cases.items():
            answer = parse_answer(text)
            self.assertEqual(answer.loose, (), name)
            self.assertTrue(answer.steps[-1].commands, name)
        loose = parse_answer("1. Update.\n2. Install it.\n\n```bash\nsudo apt install git\n```")
        self.assertEqual(loose.loose, ("sudo apt install git",))

    def test_memory_keeps_what_the_model_wrote(self):
        self.assertNotIn("```", memory_text(self.RAG_INLINE, "rag"))
        memory = memory_text(self.COT_TRAILING, "cot")
        self.assertIn("3. Set up a home directory if it is missing.\n\n```bash\nsudo adduser <username>\n```", memory)

    def test_step_note_shows_loose_commands_apart_from_the_step(self):
        note = step_note("explain step 2", self.COT_TRAILING, "cot")
        self.assertIn("step 2 was shown as:\n2. Add the user to the sudo group.\nAfter its steps it also listed", note)
        self.assertIn("say so instead of guessing", note)
        lifted = step_note("explain step 2", self.RAG_INLINE, "rag")
        self.assertIn("```bash\nusermod -aG sudo <username>\n```", lifted)

    def test_compare_step_note_labels_both_sides(self):
        note = compare_step_note("explain step 2", (
            (side_label("Left", "base", "m"), self.RAG_INLINE, "base"),
            (side_label("Right", "cot", "m"), None, "cot"),
        ))
        self.assertIn("Left answer (m · BASE): step 2 was shown as:\n2. Add the user", note)
        self.assertIn("Right answer (m · +CoT): no answer, it was stopped or failed.", note)
        self.assertEqual(compare_step_note("why sudo?", ()), "")


class StepFollowUpTests(unittest.TestCase):
    def test_step_reference(self):
        self.assertEqual(step_reference("explain step 2"), 2)
        self.assertEqual(step_reference("what does step #3 do"), 3)
        self.assertEqual(step_reference("why the second step?"), 2)
        self.assertEqual(step_reference("and the last step"), -1)
        self.assertIsNone(step_reference("explain sudo"))
        self.assertIsNone(step_reference("what is [2]"))

    def test_step_two_is_the_usermod_step_not_excerpt_two_or_reasoning(self):
        previous = memory_text("Reasoning:\n1. adduser first\n2. think about groups\n\nAnswer:\n" + SUDO, "full")
        note = step_note("explain step 2", previous)
        self.assertIn("sudo usermod -aG sudo <username>", note)
        self.assertNotIn("adduser", note)
        self.assertNotIn("think about groups", note)
        self.assertIn("not excerpt [2]", note)
        self.assertEqual(step_query("explain step 2", previous),
                         "Add the user to the sudo group: sudo usermod -aG sudo username")

    def test_last_and_missing_steps(self):
        previous = memory_text(SUDO, "rag")
        self.assertIn("groups <username>", step_note("explain the last step", previous))
        missing = step_note("explain step 5", previous)
        self.assertIn("only 3 numbered steps", missing)
        self.assertEqual(step_query("explain step 5", previous), "")
        self.assertIn("no numbered steps", step_note("explain step 1", "Use `ls`."))
        self.assertEqual(step_note("explain sudo", previous), "")


def canonical(answer):
    return "\n".join(b.text for b in answer.blocks)


if __name__ == "__main__":
    unittest.main()
