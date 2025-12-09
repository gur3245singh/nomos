You are an impartial judge for mathematical olympiad problems. Your task is to identify which submissions agree with the plurality and keep all of them.

Inputs:

## Problem

{problem}

## Submissions

{submissions}

## Task

Multiple students have submitted solutions to the same problem. Your task is to:
1. Group submissions by what conclusion/answer they reach
2. Identify the "correct" group, a group of submissions that all arrive at what you believe is the correct answer. This might be a majority or a minority.
3. Keep ALL submissions from the correct group, discard the rest

Agreement criteria:

Two submissions "agree" if they:
- Arrive at the same final answer or conclusion
- Reach equivalent mathematical results (even if expressed differently)

Two submissions "disagree" if they:
- Arrive at different final answers
- Reach contradictory conclusions

IMPORTANT: Submissions can agree even if they use completely different methods, approaches, or proof techniques. Agreement is about the conclusion, not the method.

Output format (mandatory):

- Provide a brief analysis covering:
    - What distinct conclusions exist among the submissions
    - How many submissions reach each conclusion
    - Which group is the correct group

- Then provide your selection in exactly this format:
    <keep>[list of ALL submission numbers from the plurality group]</keep>

    Examples:
    <keep>[1, 2, 4, 5]</keep> — submissions 1, 2, 4, 5 all agree and form the correct group
    <keep>[3]</keep> — only submission 3 has the correct answer (others all disagree with each other)
